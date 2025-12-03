import torch
import numpy as np
import pickle
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from torchlight import import_class

data_path = 'data_cobot_clr_zoom/xsub/train_position.npy'
label_path = 'data_cobot_clr_zoom/xsub/train_label.pkl'

data_val = np.load(data_path, mmap_mode='r')
with open(label_path, 'rb') as f:
    sample_names, labels_val = pickle.load(f)
labels_val = np.array(labels_val)

# import sys
# sys.path.append('net')

model_path = 'trial4_fixconfig_512/finetune/100%/best_model.pt'

model_class = import_class('net.aimclr_v2_3views_2.AimCLR_v2_3views')
model_args = {
    'base_encoder': 'net.ddnet.DDNet_Original',
    'pretrain': False,
    'class_num': 19,
    'frame_l': 60,
    'joint_d': 3,
    'joint_n': 48,
    'filters': 16,
    'last_feture_dim': 512,
    'feat_d': 1128

}
# model_class = import_class('net.aimclr_v2_3views.AimCLR_v2_3views')
# model_args = {
#     'base_encoder': 'net.st_gcn.Model',
#     'pretrain':False,
#     'in_channels':3,
#     'hidden_channels':32,
#     'hidden_dim':256,
#     'num_class':19,
#     'dropout':0.5,
#     'graph_args':{"layout": "cobot", "strategy": "distance"},
#     'edge_importance_weighting':True,

# }
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model_class(**model_args)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint, strict=False)
model = model.to(device)
model.eval()

import time
batch_size = 1
times = []

# Warm-up
with torch.no_grad():
    for i in range(0, len(labels_val), batch_size):
        batch_data = data_val[i:i+batch_size]
        batch_tensor = torch.from_numpy(batch_data).float().to(device)
        batch_features = model(None, batch_tensor, stream='all') 
        if device == 'cuda':
            torch.cuda.synchronize()

# Timed loop
with torch.no_grad():
    for i in range(0, len(labels_val), batch_size):
        batch_data = data_val[i:i+batch_size]
        batch_tensor = torch.from_numpy(batch_data).float().to(device)
        start = time.time()
        batch_features = model(None, batch_tensor, stream='all') 
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.time()
        
        times.append(end - start)

avg_time_ms = np.mean(times)
print(f"Average inference time per sample: {avg_time_ms:.3f} s")
