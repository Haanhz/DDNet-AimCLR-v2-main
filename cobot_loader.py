import numpy as np
from net.utils import *
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class CConfig():
    def __init__(self):
        self.frame_l = 60  # the length of frames
        self.joint_n = 48  # the number of joints
        self.joint_d = 3  # the dimension of joints
        self.clc_num = 19  # the number of class
        self.feat_d = 1128
        self.filters =16
    
    
def Cdata_generator(X, C):
    X_0 = []
    X_1 = []
    
    iu = torch.triu_indices(C.joint_n, C.joint_n, offset=1, device=X.device)
    
    for i in X:
        try:
            p = i  # shape: (frame_l, joint_n, 2)
            
            # Vectorized distance computation for all frames at once
            d_m = torch.cdist(p, p, p=2) 
            
            M = d_m[:, iu[0], iu[1]]  
            M = (M - M.mean()) / M.mean() 
            
            X_0.append(M)
            X_1.append(p)
        except Exception as e:
            print("Sample skipped:", e, " | shape:", i.shape)
            continue
    
    X_0 = torch.stack(X_0)
    X_1 = torch.stack(X_1)
    X_0 = torch.nan_to_num(X_0)
    X_1 = torch.nan_to_num(X_1)
    
    return X_0, X_1

