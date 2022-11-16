import numpy as np
from mlaas.data.semi.semi import Semi

label_cnt = 3
sample_cnt = 400
labeled_sample_cnt = 5
dim = 512

def data_generate():
    np.random.seed(100)
    
    allx = np.zeros((label_cnt*sample_cnt, dim))
    ally = np.zeros(label_cnt*sample_cnt, dtype=int)
    allgt = np.zeros(label_cnt*sample_cnt, dtype=int)
    for label in range(label_cnt):
        mean = np.random.rand(dim)/3
        var = np.random.rand(dim)
        x = np.random.normal(mean, var, (sample_cnt, dim))
        y = np.ones(sample_cnt, dtype=int)*-1
        y[np.random.choice(sample_cnt, labeled_sample_cnt, False)] = label
        gt = np.ones(sample_cnt, dtype=int)*label
        
        allx[label*sample_cnt:(label+1)*sample_cnt] = x
        ally[label*sample_cnt:(label+1)*sample_cnt] = y
        allgt[label*sample_cnt:(label+1)*sample_cnt] = gt
    
    return allx, ally, allgt

if __name__ == '__main__':
    x, y, gt = data_generate()
    semi = Semi(x, y, gt=gt, n_neighbor=5)
    # init training
    semi.fit()
    
    # add labeled data
    for label in range(label_cnt):
        relabel_idx = np.random.choice(sample_cnt, 2, False)+label*sample_cnt
        y[relabel_idx] = gt[relabel_idx]
    semi.fit()
    
    # local change k
    idxs = list(range(600, 650))
    k_candidates = list(range(10, 20))
    semi.local_change_k(idxs, k_candidates)
    semi.fit()