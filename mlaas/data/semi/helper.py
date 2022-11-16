import warnings

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.stats import entropy
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import paired_distances

def build_laplacian_graph(affinity_matrix):
    instance_num = affinity_matrix.shape[0]
    laplacian = csgraph.laplacian(affinity_matrix, normed=True)
    laplacian = -laplacian
    if sparse.isspmatrix(laplacian):
        diag_mask = (laplacian.row == laplacian.col)
        laplacian.data[diag_mask] = 0.0
    else:
        laplacian.flat[::instance_num + 1] = 0.0  # set diag to 0.0
    return laplacian

def propagation(graph_matrix, affinity_matrix, train_y, alpha=0.2, max_iter=15,
                tol=1e-12, process_record=False, normalized=False, k=6):
    y = np.array(train_y)
    # label construction
    # construct a categorical distribution for classification only
    classes = np.unique(y)
    classes = (classes[classes != -1])
    process_data = None

    D = affinity_matrix.sum(axis=0).getA1() - affinity_matrix.diagonal()
    D = np.sqrt(D)
    D[D == 0] = 1
    affinity_matrix.setdiag(0)

    n_samples, n_classes = len(y), len(classes)

    if (alpha is None or alpha <= 0.0 or alpha >= 1.0):
        raise ValueError('alpha=%s is invalid: it must be inside '
                         'the open interval (0, 1)' % alpha)
    y = np.asarray(y)
    unlabeled = y == -1
    labeled = (y > -1)

    # initialize distributions
    label_distributions_ = np.zeros((n_samples, n_classes))
    for label in classes:
        label_distributions_[y == label, classes == label] = 1

    y_static_labeled = np.copy(label_distributions_)
    y_static = y_static_labeled * (1 - alpha)

    l_previous = np.zeros((n_samples, n_classes))

    unlabeled = unlabeled[:, np.newaxis]
    if sparse.isspmatrix(graph_matrix):
        graph_matrix = graph_matrix.tocsr()

    all_loss = []
    all_entropy = []

    if process_record:
        label = label_distributions_.copy()
        if normalized:
            normalizer = np.sum(label, axis=1)[:, np.newaxis]
            normalizer = normalizer + 1e-20
            label /= normalizer
        process_data = [label]
        ent = entropy(label.T + 1e-20)
        all_entropy.append(ent.sum())

    n_iter_ = 1
    for _ in range(max_iter):
        if np.abs(label_distributions_ - l_previous).sum() < tol:
                break

        l_previous = label_distributions_.copy()
        label_distributions_a = safe_sparse_dot(
            graph_matrix, label_distributions_)

        if not (n_iter_ > 6 and k <= 3): # for case
            label_distributions_ = np.multiply(
                alpha, label_distributions_a) + y_static
        n_iter_ += 1
        if process_record:
            label = label_distributions_.copy()
            if normalized:
                normalizer = np.sum(label, axis=1)[:, np.newaxis]
                normalizer = normalizer + 1e-20
                label /= normalizer
            process_data.append(label)
            ent = entropy(label.T + 1e-20)
            all_entropy.append(ent.sum())

        # record loss
        t = ((l_previous / D[:, np.newaxis]) ** 2).sum(axis=1)
        loss = safe_sparse_dot(affinity_matrix.sum(axis=1).T, t) * 0.5 + \
               safe_sparse_dot(affinity_matrix.sum(axis=0), t) * 0.5 - \
               np.dot(l_previous.reshape(-1),
                      label_distributions_a.reshape(-1))
        # loss[0, 0]: read the only-one value in a numpy.matrix variable
        loss = loss[0, 0] + alpha / (1 - alpha) * paired_distances(label_distributions_[labeled],
                                                                   y_static_labeled[labeled]).sum()
        all_loss.append(loss)

    else:
        warnings.warn(
            'max_iter=%d was reached without convergence.' % max_iter,
            category=ConvergenceWarning
        )
        # n_iter_ += 1

    unnorm_dist = label_distributions_.copy()

    if normalized:
        normalizer = np.sum(label_distributions_, axis=1)[:, np.newaxis]
        normalizer = normalizer + 1e-20
        label_distributions_ /= normalizer

    all_loss.append(all_loss[-1])
    all_loss = np.array(all_loss)
    all_entropy = np.array(all_entropy)
    assert np.isnan(all_entropy).sum() == 0
    assert np.isinf(all_entropy).sum() == 0

    if process_data is not None:
        process_data = np.array(process_data)

        labels = process_data.argmax(axis=2)
        max_process_data = process_data.max(axis=2)
        labels[max_process_data == 0] = -1

        # remove unnecessary iterations
        assert n_iter_ == len(process_data), "{}, {}".format(n_iter_, len(process_data))
        new_iter_num = n_iter_ - 1
        if not (n_iter_ > 6 and k <= 3): # for case
            for new_iter_num in range(n_iter_ - 1, 0, -1):
                if sum(labels[new_iter_num - 1] != labels[n_iter_- 1]) != 0:
                    break

        process_data[new_iter_num] = process_data[n_iter_ - 1]
        process_data = process_data[:new_iter_num + 1]
        all_loss[new_iter_num] = all_loss[n_iter_ - 1]
        all_loss = all_loss[:new_iter_num + 1]
        all_entropy[new_iter_num] = all_entropy[n_iter_ - 1]
        all_entropy = all_entropy[:new_iter_num + 1]

    return label_distributions_, all_loss, all_entropy, process_data, unnorm_dist

def pure_propagation(graph_matrix, affinity_matrix, train_y, alpha=0.2, max_iter=30,
                tol=1e-12, process_record=False, normalized=False):
    y = np.array(train_y)
    # label construction
    # construct a categorical distribution for classification only
    classes = np.unique(y)
    classes = (classes[classes != -1])
    process_data = None

    # affinity_matrix.setdiag(0)

    n_samples, n_classes = len(y), len(classes)

    if (alpha is None or alpha <= 0.0 or alpha >= 1.0):
        raise ValueError('alpha=%s is invalid: it must be inside '
                         'the open interval (0, 1)' % alpha)
    y = np.asarray(y)
    unlabeled = y == -1
    labeled = (y > -1)

    # initialize distributions
    label_distributions_ = np.zeros((n_samples, n_classes))
    for label in classes:
        label_distributions_[y == label, classes == label] = 1

    y_static_labeled = np.copy(label_distributions_)
    y_static = y_static_labeled * (1 - alpha)

    l_previous = np.zeros((n_samples, n_classes))

    if sparse.isspmatrix(graph_matrix):
        graph_matrix = graph_matrix.tocsr()



    n_iter_ = 1
    for _ in range(max_iter):
        if np.abs(label_distributions_ - l_previous).sum() < tol:
            break

        l_previous = label_distributions_.copy()
        label_distributions_a = safe_sparse_dot(
            graph_matrix, label_distributions_)

        label_distributions_ = np.multiply(
            alpha, label_distributions_a) + y_static
        n_iter_ += 1

    else:
        warnings.warn(
            'max_iter=%d was reached without convergence.' % max_iter,
            category=ConvergenceWarning
        )
        n_iter_ += 1

    unnorm_dist = label_distributions_.copy()

    if normalized:
        normalizer = np.sum(label_distributions_, axis=1)[:, np.newaxis]
        normalizer = normalizer + 1e-20
        label_distributions_ /= normalizer



    return label_distributions_, unnorm_dist


def full_update(selected_idxs, F, graph_matrix, affinity_matrix, train_y, alpha=0.2, max_iter=30,
                tol=0.001, process_record=False, normalized=False):
        pred_dist, unnorm_dist = \
            pure_propagation(graph_matrix, affinity_matrix, train_y,
                              alpha=alpha, process_record=True,
                              normalized=True)
        return pred_dist, 1

def local_search_k(k_list, n_neighbors, selected_idxs, F, initial_affinity_matrix,
    train_y, neighbors, gt = None):
    print("selected_idxs len:", len(selected_idxs))
    normalizer = np.sum(F, axis=1)[:, np.newaxis] + 1e-20
    norm_F = F / normalizer
    original_ent = entropy(norm_F.T + 1e-20).mean()
    best_affinity_matrix = None
    min_ent = original_ent + 2000
    best_k = None
    best_affinity_matrix = initial_affinity_matrix.copy()
    best_pred = None
    selected_num = len(selected_idxs)
    instance_num = len(train_y)

    unselected_idxs = np.ones(instance_num).astype(bool)
    unselected_idxs[selected_idxs] = False
    unselected_idxs = np.array(range(instance_num))[unselected_idxs]

    for local_k in k_list:
        if local_k <= 1:
            continue
        indptr = [i * local_k for i in range(selected_num + 1)]
        indices = neighbors[selected_idxs][:, :local_k].reshape(-1).tolist()
        data = neighbors[selected_idxs][:, :local_k].reshape(-1)
        data = (data * 0 + 1.0).tolist()
        selected_affinity_matrix = sparse.csr_matrix((data, indices, indptr),
            shape=(selected_num, instance_num)).toarray()
        affinity_matrix = initial_affinity_matrix.toarray()
        affinity_matrix[:, selected_idxs] = selected_affinity_matrix.T
        affinity_matrix[selected_idxs, :] = selected_affinity_matrix
        affinity_matrix = sparse.csr_matrix(affinity_matrix)
        affinity_matrix.setdiag(0)

        laplacian_matrix = build_laplacian_graph(affinity_matrix)
        pred, iter_num = full_update(selected_idxs, F, laplacian_matrix, affinity_matrix,
            train_y, normalized=True)
        if gt is not None:
            acc = accuracy_score(gt[selected_idxs], pred.argmax(axis=1)[selected_idxs])
        else:
            acc = 'unknown'
        max_pred = pred.max(axis=1)
        prop_pred = pred[max_pred != 0]
        ent = entropy(prop_pred.T + 1e-20).mean()
        # print(local_k, acc, "ent:", ent, min_ent, iter_num)
        if ent < min_ent:
            print("update k:", ent, min_ent)
            min_ent = ent
            best_k = local_k
            best_affinity_matrix = affinity_matrix
            best_pred = pred
    # print("best local k:", best_k, "best_ent", min_ent, "original_ent", original_ent)
    return best_affinity_matrix, best_pred, best_k