from .earlytrain import EarlyTrain
import torch
import numpy as np
from ..nets.nets_utils import MyDataParallel
from .methods_utils import euclidean_dist, PCAcomponent_1, PCAcomponent
from collections import defaultdict
from tqdm import tqdm

def index_new2standard(new_index_list, standard_index_list):
    change_result = []
    for i in range(len(new_index_list)):
        sub_list = new_index_list[i]
        change_sub_list = []
        for j in range(len(sub_list)):
            old_index = sub_list[j]
            standard_index = standard_index_list[old_index]
            change_sub_list.append(standard_index)
        change_result.append(change_sub_list)

    return change_result

def CDS_constraint_function(feat_center, feat_list, index_list, budget):
    sample_num = len(index_list)

    feat_matrix = torch.cat(feat_list, dim=0)
    dist_dim_matrix = feat_center - feat_matrix
    dist_dim_matrix = torch.abs(dist_dim_matrix)

    beta = 1e-1 #Hyper-parameter
    mask = dist_dim_matrix.gt(beta)

    CDS_cluster_list = []
    budget_list = []
    homogeny_cluster = defaultdict(list)
    index_feat = np.arange(sample_num)
    index_mask = np.zeros(sample_num, dtype=bool)
    for i in range(sample_num):
        if index_mask[i]:
            continue
        homogeny_cluster[i].append(i)
        index_mask[i] = True
        for _, j in enumerate(index_feat[~index_mask]):
            flag = (~torch.eq(mask[i], mask[j])).sum().item()
            if flag == 0:
                homogeny_cluster[i].append(j)
                index_mask[j] = True
    cluster_num = len(homogeny_cluster.keys())
    sort_max2min = sorted(homogeny_cluster.items(), key = lambda x:len(x[1]), reverse = True)
    num_selected = 0
    if budget <= cluster_num:
        for i in range(len(sort_max2min)):
            index_candidate = sort_max2min[i][1]
            CDS_cluster_list.append(index_candidate)
            budget_list.append(1)
            num_selected += 1
            if num_selected == budget:
                break
    else: 
        num_2select_list = [0]*cluster_num
        complete_num = 0
        num_2divide = budget
        while True:
            if num_2select_list[-complete_num-1] < len(sort_max2min[-complete_num-1][1]):
                for j in range(cluster_num-complete_num):
                    num_2select_list[j] += 1
                    num_2divide -= 1
                    if num_2divide == 0:
                        break
                    if num_2select_list[j] == len(sort_max2min[j][1]):
                        complete_num += 1
            else:
                complete_num += 1

            if num_2divide == 0:
                break

        for i in range(len(sort_max2min)):
            index_candidate = sort_max2min[i][1]
            num_2select = num_2select_list[i]
            CDS_cluster_list.append(index_candidate)
            budget_list.append(num_2select)

    CDS_cluster_list = index_new2standard(CDS_cluster_list, index_list)

    return CDS_cluster_list, budget_list

# 1st Clustring + 2nd Clustering
def Hard_CDS_function(matrix, budget: int, metric, device, random_seed=None, index=None):
    if type(matrix) == torch.Tensor:
        assert matrix.dim() == 2
    elif type(matrix) == np.ndarray:
        assert matrix.ndim == 2
        matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)

    sample_num = matrix.shape[0]
    assert sample_num >= 1

    if budget < 0:
        raise ValueError("Illegal budget size.")
    elif budget > sample_num:
        budget = sample_num

    if index is not None:
        assert matrix.shape[0] == len(index)
    else:
        index = np.arange(sample_num)

    assert callable(metric)

    cluster_list = []
    budget_list = []
    with torch.no_grad():
        np.random.seed(random_seed)

        feat_center =  torch.mean(matrix, axis=0)
        feat_center = feat_center.unsqueeze(axis=0)

        dis_matrix = metric(feat_center, matrix)
        feat_cluster = defaultdict(list)
        index_cluster = defaultdict(list)
        num_candidate = sample_num

        for i in range(sample_num):
            alpha = 0.5 #Hyper-parameter
            cluster_indx = torch.div(dis_matrix[0][i], alpha, rounding_mode='floor').item()
            feat_cluster[cluster_indx].append(matrix[i].unsqueeze(axis=0))
            index_cluster[cluster_indx].append(i)

        num_selected = 0
        sort_min2max = sorted(index_cluster.items(), key = lambda x:len(x[1]))
        end_k = sort_min2max[-1][0]

        for i in tqdm(range(len(sort_min2max))):
            k = sort_min2max[i][0]

            if k == end_k:
                num_2select = budget - num_selected
            else:
                num_2select = round((len(index_cluster[k])/num_candidate)*budget)
            
            if num_2select > len(index_cluster[k]):
                num_2select = len(index_cluster[k])

            if num_2select == len(index_cluster[k]):
                cluster_list.append(index_cluster[k])
                budget_list.append(num_2select)
            elif num_2select:
                index_selected_list, sub_budget_list = CDS_constraint_function(feat_center, feat_cluster[k], index_cluster[k], num_2select)
                cluster_list.extend(index_selected_list)
                budget_list.extend(sub_budget_list)

            num_selected += num_2select

    cluster_list = index_new2standard(cluster_list, index.tolist())

    return cluster_list, budget_list

def dimension_reduction(feat_matrix, device):
    if type(feat_matrix) == torch.Tensor:
        assert feat_matrix.dim() == 2
        feat_matrix = feat_matrix.cpu().numpy()
    elif type(feat_matrix) == np.ndarray:
        assert feat_matrix.ndim == 2
    feat_matrix = np.mat(feat_matrix)
    dim_reduce_k = 10 #Hyper-parameter

    pca = PCAcomponent(feat_matrix, dim_reduce_k)
    pca.fit()
    matrix = pca.low_dataMat
    matrix = matrix.real
    matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)

    return matrix

class Uncertainty_Hard(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, selection_method="LeastConfidence",
                 specific_model=None, balance=False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model, **kwargs)

        selection_choices = ["LeastConfidence",
                             "Entropy",
                             "Margin"]
        if selection_method not in selection_choices:
            raise NotImplementedError("Selection algorithm unavailable.")
        self.selection_method = selection_method

        self.epochs = epochs
        self.balance = balance

    def before_train(self):
        pass

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_run(self):
        pass

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
            epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))

    def construct_matrix(self, index=None):
        self.model.eval()
        self.model.no_grad = True
        with torch.no_grad():
            with self.model.embedding_recorder:
                matrix = []

                data_loader = torch.utils.data.DataLoader(self.dst_train if index is None else
                                    torch.utils.data.Subset(self.dst_train, index),
                                    batch_size=self.args.selection_batch,
                                    num_workers=self.args.workers)

                for i, (inputs, _) in enumerate(data_loader):
                    self.model(inputs.to(self.args.device))
                    matrix.append(self.model.embedding_recorder.embedding)

        self.model.no_grad = False
        return torch.cat(matrix, dim=0)

    def rank_uncertainty(self, index=None):
        self.model.eval()
        with torch.no_grad():
            train_loader = torch.utils.data.DataLoader(
                self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
                batch_size=self.args.selection_batch,
                num_workers=self.args.workers)

            scores = np.array([])
            batch_num = len(train_loader)

            print('self.selection_method=', self.selection_method)
            for i, (input, _) in enumerate(train_loader):
                if i % self.args.print_freq == 0:
                    print("| Selecting for batch [%3d/%3d]" % (i + 1, batch_num))
                if self.selection_method == "LeastConfidence":
                    scores = np.append(scores, self.model(input.to(self.args.device)).max(axis=1).values.cpu().numpy())
                elif self.selection_method == "Entropy":
                    preds = torch.nn.functional.softmax(self.model(input.to(self.args.device)), dim=1).cpu().numpy()
                    scores = np.append(scores, (np.log(preds + 1e-6) * preds).sum(axis=1))
                elif self.selection_method == 'Margin':
                    preds = torch.nn.functional.softmax(self.model(input.to(self.args.device)), dim=1)
                    preds_argmax = torch.argmax(preds, dim=1)
                    max_preds = preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax].clone()
                    preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax] = -1.0
                    preds_sub_argmax = torch.argmax(preds, dim=1)
                    scores = np.append(scores, (max_preds - preds[
                        torch.ones(preds.shape[0], dtype=bool), preds_sub_argmax]).cpu().numpy())
        return scores

    def finish_run(self):
        scores = self.rank_uncertainty() #np.array

        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module
        if self.balance:
            selection_result = np.array([], dtype=np.int64)
            scores_list = []
            for c in range(self.args.num_classes):
                class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                scores_c = scores[self.dst_train.targets == c]

                feature_matrix = self.construct_matrix(class_index)
                feature_matrix = dimension_reduction(feat_matrix=feature_matrix, device=self.args.device)
                c_indx_list, budget_list = Hard_CDS_function(feature_matrix, 
                                budget=round(self.fraction * len(class_index)),
                                metric=euclidean_dist,
                                device=self.args.device,
                                random_seed=self.random_seed,
                                index=class_index)

                for i in range(len(budget_list)):
                    budget = budget_list[i]
                    c_indx = c_indx_list[i]
                    c_indx = np.array(c_indx)

                    if budget == len(c_indx):
                        class_result = c_indx
                    elif budget:
                        c_mask = np.isin(class_index, c_indx)
                        scores_list.append(scores_c[c_mask])
                        class_result = c_indx[np.argsort(scores_list[-1])[:budget]]
                    selection_result = np.append(selection_result, class_result)
        else:
            print('to be done')
        return {"indices": selection_result}

    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result
