import numpy as np
from mlaas.data.ood.ood import OodDetection

label_cnt = 5
sample_cnt = 400
test_cnt = 50
dim = 512

def data_generate():
    np.random.seed(100)
    
    allx = np.zeros((label_cnt*sample_cnt, dim))
    ally = np.zeros(label_cnt*sample_cnt, dtype=int)
    testx = np.zeros((label_cnt*test_cnt, dim))

    for label in range(label_cnt):
        mean = np.random.rand(dim)/3
        var = np.random.rand(dim)
        x = np.random.normal(mean, var, (sample_cnt, dim))
        y = np.ones(sample_cnt, dtype=int)*label
        tx = np.random.normal(mean, var, (test_cnt, dim))
        
        allx[label*sample_cnt:(label+1)*sample_cnt] = x
        ally[label*sample_cnt:(label+1)*sample_cnt] = y
        testx[label*test_cnt:(label+1)*test_cnt] = tx
    
    return allx, ally, testx

if __name__ == '__main__':
    x, y, tx = data_generate()
    oodDetector = OodDetection()
    
    # fit ood score
    oodScore = oodDetector.fit(x, y, tx)
    print(oodScore)