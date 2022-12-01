import numpy as np
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression

class OodDetection(object):
    def __init__(self, candidates = None) -> None:
        self.candidates = candidates
        if self.candidates is None:
            self.candidates = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
            
        self.clfs = []
        for C in self.candidates:
            # for C in [1e-2, 1e-1, 1, 1e1]:
                self.clfs.append(LogisticRegression(C=C))
                
    def fit(self, trainX, trainY, testX):
        test_predys = None
        
        for clf in self.clfs:
            clf.fit(trainX, trainY)
            test_predy = clf.predict_proba(testX)
            if test_predys is None:
                test_predys = test_predy
            else:
                test_predys = test_predys + test_predy
        return entropy(test_predys.T)
        
        