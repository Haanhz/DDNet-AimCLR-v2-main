
# from tqdm import tqdm
# from sklearn import preprocessing
# from pathlib import Path
# import sys

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
    X_0=[]
    X_1=[]

    for i in X:
        try:
            i = i.detach().cpu().numpy()
            p = np.copy(i)
            #p = zoom(p, target_l=C.frame_l, joints_num=C.joint_n, joints_dim=C.joint_d)
            M = get_CG(p, C)
            X_0.append(M)
            X_1.append(p)
        except Exception as e:
            print("Sample skipped:", e, " | shape:", i.shape)
            continue

    X_0 = np.stack(X_0)
    X_1 = np.stack(X_1)
    X_0 = np.nan_to_num(X_0)
    X_1 = np.nan_to_num(X_1)
    X_0 = torch.tensor(X_0, dtype=torch.float32).to(device)
    X_1 = torch.tensor(X_1, dtype=torch.float32).to(device)
    return X_0, X_1

