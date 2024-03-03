from .earlytrain import EarlyTrain
import torch
import numpy as np
from .methods_utils import euclidean_dist, cossim_np, cossim_pair_np, submodular_function, submodular_optimizer, PCAcomponent_1, PCAcomponent
from ..nets.nets_utils import MyDataParallel
import time
    
def CDS_metric_function(matrix, device):
    if type(matrix) == torch.Tensor:
        assert matrix.dim() == 2
    elif type(matrix) == np.ndarray:
        assert matrix.ndim == 2
        matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)
    
    central_feature =  torch.mean(matrix, axis=0)
    central_feature = central_feature.unsqueeze(axis=0)

    dist_dim_matrix = central_feature - matrix
    dist_dim_matrix = torch.abs(dist_dim_matrix)

    beta = 1e-4 #Hyper-parameter
    CDS = dist_dim_matrix.gt(beta).cpu().numpy()
    CDS = CDS.astype(int)

    CDS_relationship = cossim_pair_np(CDS)
    CDS_relationship_mask = CDS_relationship.__gt__(0.999999999)
    CDS_relationship_mask = CDS_relationship_mask.astype(int)

    constraint_matrix = CDS_relationship_mask + np.ones([len(CDS_relationship_mask[0]), len(CDS_relationship_mask[0])], dtype=np.float32)

    return constraint_matrix

def dimension_reduction(feat_matrix, device):
    if type(feat_matrix) == torch.Tensor:
        assert feat_matrix.dim() == 2
        feat_matrix = feat_matrix.cpu().numpy()
    elif type(feat_matrix) == np.ndarray:
        assert feat_matrix.ndim == 2
    feat_matrix = np.mat(feat_matrix)
    dim_reduce_k = 10 #Hyper-parameter

    pca = PCAcomponent_1(feat_matrix, dim_reduce_k)
    pca.fit()
    matrix = pca.low_dataMat
    matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)

    return matrix

class Submodular_Soft(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200,
                 specific_model="ResNet18", balance: bool = False, already_selected=[], metric="euclidean",
                 torchvision_pretrain: bool = True, function="LogDeterminant", greedy="ApproximateLazyGreedy", 
                 _metric="cossim", **kwargs):
        super(Submodular_Soft, self).__init__(dst_train, args, fraction, random_seed, epochs=epochs, specific_model=specific_model,
                         torchvision_pretrain=torchvision_pretrain, **kwargs)

        if already_selected.__len__() != 0:
            if min(already_selected) < 0 or max(already_selected) >= self.n_train:
                raise ValueError("List of already selected points out of the boundary.")
        self.already_selected = np.array(already_selected)

        self.min_distances = None

        if metric == "euclidean":
            self.metric = euclidean_dist
        elif callable(metric):
            self.metric = metric
        else:
            self.metric = euclidean_dist
            self.run = lambda : self.finish_run()
            def _construct_matrix(index=None):
                data_loader = torch.utils.data.DataLoader(
                    self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
                    batch_size=self.n_train if index is None else len(index),
                    num_workers=self.args.workers)
                inputs, _ = next(iter(data_loader))
                return inputs.flatten(1).requires_grad_(False).to(self.args.device)
            self.construct_matrix = _construct_matrix
        
        if greedy not in submodular_optimizer.optimizer_choices:
            raise ModuleNotFoundError("Greedy optimizer not found.")
        self._greedy = greedy
        self._metric = _metric #used for submodular
        self._function = function #GraphCut

        self.balance = balance

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
            epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))

    def construct_matrix(self, index=None):
        self.model.eval()
        with torch.no_grad():
            matrix = []

            data_loader = torch.utils.data.DataLoader(self.dst_train if index is None else
                                torch.utils.data.Subset(self.dst_train, index),
                                batch_size=self.args.selection_batch,
                                num_workers=self.args.workers)

            for i, (inputs, targets) in enumerate(data_loader):
                self.model(inputs.to(self.args.device))
                matrix.append(self.model.embedding_recorder.embedding)

        return torch.cat(matrix, dim=0)
    
    def calc_gradient(self, index=None):
        '''
        Calculate gradients matrix on current network for specified training dataset.
        '''
        self.model.eval()

        batch_loader = torch.utils.data.DataLoader(
                self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
                batch_size=self.args.selection_batch,
                num_workers=self.args.workers)

        # Initialize a matrix to save gradients.
        # (on cpu)
        gradients = []

        for i, (input, targets) in enumerate(batch_loader):
            self.model_optimizer.zero_grad()
            outputs = self.model(input.to(self.args.device))
            loss = self.criterion(outputs.requires_grad_(True),
                                  targets.to(self.args.device)).sum()
            batch_num = targets.shape[0]
            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, outputs)[0]
                weight_parameters_grads = self.model.embedding_recorder.embedding.view(batch_num, 1,
                                        self.emb_dim).repeat(1, self.args.num_classes, 1) *\
                                        bias_parameters_grads.view(batch_num, self.args.num_classes,
                                        1).repeat(1, 1, self.emb_dim)
                gradients.append(torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)],
                                            dim=1).cpu().numpy())

        gradients = np.concatenate(gradients, axis=0)
        return gradients

    def get_hms(self, seconds):
        # Format time for printing purposes

        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return h, m, s

    def before_epoch(self):
        self.start_time = time.time()

    def after_epoch(self):
        epoch_time = time.time() - self.start_time
        self.elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (self.get_hms(self.elapsed_time)))

    def before_run(self):
        self.elapsed_time = 0
        self.emb_dim = self.model.get_last_layer().in_features

    def finish_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module

    def select(self, **kwargs):
        self.run()
        with self.model.embedding_recorder:
            self.model.no_grad = True
            self.train_indx = np.arange(self.n_train)

            if self.balance:
                selection_result = np.array([], dtype=np.int32)
                for c in range(self.num_classes):
                    c_indx = self.train_indx[self.dst_train.targets == c]

                    feature_matrix = self.construct_matrix(c_indx)
                    feature_matrix= dimension_reduction(feat_matrix=feature_matrix, device=self.args.device)
                    constraint_matrix = CDS_metric_function(matrix=feature_matrix, device=self.args.device)

                    # Calculate gradients into a matrix
                    gradients = self.calc_gradient(index=c_indx)
                    # Instantiate a submodular function
                    submod_function = submodular_function.__dict__[self._function](constraint_matrix=constraint_matrix, index=c_indx,
                                    similarity_kernel=lambda a, b:cossim_np(gradients[a], gradients[b]))
                    submod_optimizer = submodular_optimizer.__dict__[self._greedy](args=self.args,
                                    index=c_indx, budget=round(self.fraction * len(c_indx)), already_selected=[])

                    c_selection_result = submod_optimizer.select(gain_function=submod_function.calc_gain,
                                                            update_state=submod_function.update_state)

                    selection_result = np.append(selection_result, c_selection_result)
            else:
                feature_matrix = self.construct_matrix()
                feature_matrix = dimension_reduction(feat_matrix=feature_matrix, device=self.args.device)
                constraint_matrix = CDS_metric_function(matrix=feature_matrix, device=self.args.device)

                # Calculate gradients into a matrix
                gradients = self.calc_gradient()
                # Instantiate a submodular function
                submod_function = submodular_function.__dict__[self._function](constraint_matrix=constraint_matrix, index=self.train_indx,
                                            similarity_kernel=lambda a, b: cossim_np(gradients[a], gradients[b]))
                submod_optimizer = submodular_optimizer.__dict__[self._greedy](args=self.args, index=self.train_indx,
                                                                                  budget=self.coreset_size)
                selection_result = submod_optimizer.select(gain_function=submod_function.calc_gain,
                                                           update_state=submod_function.update_state)

            self.model.no_grad = False

        return {"indices": selection_result}
