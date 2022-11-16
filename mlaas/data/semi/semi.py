import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score

from mlaas.data.semi.helper import build_laplacian_graph, propagation, local_search_k

class Semi(object):
    def __init__(self, X, Y, gt = None, n_neighbor = 5) -> None:
        self.X = X
        self.Y = Y
        self.gt = gt
        self.n_neighbor = n_neighbor
        
        self.max_neighbors = 100
        self.alpha = 0.2
        
        # computed in self._preprocess_neighbors()
        self.neighbors = None
        self.neighbors_weight = None
        # computed in self.affinity_matrix()
        self.affinity_matrix = None
        # computed in self.fit()
        self.labels = None
        self.unnorm_dist = None
        
    def csr_to_impact_matrix(self, neighbor_result, instance_num, max_neighbors):
        neighbors = np.zeros((instance_num, max_neighbors)).astype(int)
        neighbors_weight = np.zeros((instance_num, max_neighbors))
        for i in range(instance_num):
            start = neighbor_result.indptr[i]
            end = neighbor_result.indptr[i + 1]
            j_in_this_row = neighbor_result.indices[start:end]
            data_in_this_row = neighbor_result.data[start:end]
            sorted_idx = data_in_this_row.argsort()
            assert (len(sorted_idx) == max_neighbors)
            j_in_this_row = j_in_this_row[sorted_idx]
            data_in_this_row = data_in_this_row[sorted_idx]
            neighbors[i, :] = j_in_this_row
            neighbors_weight[i, :] = data_in_this_row
        return neighbors, neighbors_weight
        
    def _preprocess_neighbors(self):
        train_num = len(self.X)
        
        nn_fit = NearestNeighbors(n_neighbors=10, n_jobs=-1).fit(self.X)
        neighbor_result = nn_fit.kneighbors_graph(nn_fit._fit_X, self.max_neighbors, mode="distance")
        self.neighbors, self.neighbors_weight = self.csr_to_impact_matrix(neighbor_result, train_num, self.max_neighbors)
        
    def _construct_graph(self):
        if self.neighbors is None or self.neighbors_weight is None:
            self._preprocess_neighbors()
        neighbors = self.neighbors
        neighbors_weight = self.neighbors_weight
        n_neighbor = self.n_neighbor
        instance_num = len(self.X)
        
        # get knn graph in a csr form
        indptr = [i * n_neighbor for i in range(instance_num + 1)]
        indices = neighbors[:, :n_neighbor].reshape(-1).tolist()
        weight = False
        if not weight:
            data = neighbors[:, :n_neighbor].reshape(-1)
            data = (data * 0 + 1.0).tolist()
        else:
            data = neighbors_weight[:, :n_neighbor].reshape(-1).tolist()
        affinity_matrix = sparse.csr_matrix((data, indices, indptr),
                                            shape=(instance_num, instance_num))
        affinity_matrix = affinity_matrix + affinity_matrix.T
        affinity_matrix = sparse.csr_matrix((np.ones(len(affinity_matrix.data)).tolist(),
                                             affinity_matrix.indices, affinity_matrix.indptr),
                                            shape=(instance_num, instance_num))

        self.affinity_matrix = affinity_matrix

        return affinity_matrix
    
    def fit(self):
        if self.affinity_matrix is None:
            self._construct_graph()
        affinity_matrix = self.affinity_matrix
        laplacian = build_laplacian_graph(affinity_matrix)
        
        pred_dist, loss, ent, process_data, unnorm_dist = propagation(laplacian, affinity_matrix, self.Y,
                              alpha=self.alpha, process_record=True,
                              normalized=True, k=self.n_neighbor)
        
        # get labels and flows
        self.unnorm_dist = unnorm_dist
        self.labels = process_data.argmax(axis=2)
        max_process_data = process_data.max(axis=2)
        self.labels[max_process_data == 0] = -1
        print("unpropagated instance num: {}".format(sum(self.labels[-1]==-1)))
        if self.gt is not None:
            pred_y = pred_dist.argmax(axis=1)
            acc = accuracy_score(self.gt, pred_y)
            print("model accuracy: {}".format(acc))
        
    def label_instance(self, idxs, labels):
        for i in range(len(idxs)):
            idx = idxs[i]
            label = labels[i]
            self.Y[idx] = label
            
    def add_edge(self, removed_edges):
        for edges in removed_edges:
            s, e = edges
            self.affinity_matrix[s, e] = 1
            self.affinity_matrix[e, s] = 1

    def remove_edge(self, removed_edges):
        for edges in removed_edges:
            s, e = edges
            self.affinity_matrix[s, e] = 0
            self.affinity_matrix[e, s] = 0
            
    def local_change_k(self, selected_idxs, k_candidates = None):
        if k_candidates is None:
            k_candidates = list(range(1, 16))
        
        unnorm_dist = self.unnorm_dist
        affinity_matrix = self.affinity_matrix
        affinity_matrix.setdiag(0)
        affinity_matrix, pred, best_k = local_search_k(k_candidates, self.n_neighbor,
            selected_idxs, unnorm_dist, affinity_matrix, 
            self.Y, self.neighbors, self.gt)
        self.affinity_matrix = self.correct_unconnected_nodes(affinity_matrix)
    
    def _find_unconnected_nodes(self, affinity_matrix, labeled_id):
        edge_indices = affinity_matrix.indices
        edge_indptr = affinity_matrix.indptr
        node_num = edge_indptr.shape[0] - 1
        connected_nodes = np.zeros((node_num))
        connected_nodes[labeled_id] = 1

        iter_cnt = 0
        while True:
            new_connected_nodes = affinity_matrix.dot(connected_nodes)+connected_nodes
            new_connected_nodes = new_connected_nodes.clip(0, 1)
            iter_cnt += 1
            if np.allclose(new_connected_nodes, connected_nodes):
                break
            connected_nodes = new_connected_nodes
        unconnected_nodes = np.where(new_connected_nodes<1)[0]
        return unconnected_nodes
        
    def correct_unconnected_nodes(self, affinity_matrix):
        np.random.seed(123)
        correted_nodes = []
        affinity_matrix = affinity_matrix.copy()
        labeled_ids = np.where(self.Y > -1)[0]
        iter_cnt = 0
        neighbors = self.neighbors
        while True:
            unconnected_ids = self._find_unconnected_nodes(affinity_matrix, labeled_ids)
            if unconnected_ids.shape[0] == 0:
                print(f'correct {len(correted_nodes)} nodes: {correted_nodes}')
                return affinity_matrix
            else:
                while True:
                    corrected_id = np.random.choice(unconnected_ids)
                    k_neighbors = neighbors[corrected_id]
                    find = False
                    for neighbor_id in k_neighbors:
                        if neighbor_id not in unconnected_ids:
                            find = True
                            iter_cnt += 1
                            affinity_matrix[corrected_id, neighbor_id] = 1
                            correted_nodes.append([corrected_id, neighbor_id])
                            break
                    if find:
                        break