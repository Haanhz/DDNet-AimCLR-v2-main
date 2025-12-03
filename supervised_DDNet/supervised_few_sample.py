import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Lambda,
    Reshape,
    Conv1D,
    BatchNormalization,
    LeakyReLU,
    SpatialDropout1D,
    MaxPooling1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout,
    concatenate
)

from utils import *
from tensorflow.keras.models import Model
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import random
import pickle
# from tqdm import tqdm
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import time
from tensorflow.keras.callbacks import ModelCheckpoint

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Optional: Print GPU details
if tf.config.list_physical_devices('GPU'):
    print("GPU details:", tf.config.list_physical_devices('GPU'))
gpus = tf.config.list_physical_devices('GPU')

tf.config.set_visible_devices(gpus[2], 'GPU')
print("Using GPU:", gpus[2])

random.seed(123)

class Config():
    def __init__(self):
        self.frame_l = 60 # the length of frames (avg is 93.55)
        self.joint_n = 48 # the number of joints
        self.joint_d = 3 # the dimension of joints
        self.clc_num = 19 # the number of class
        self.feat_d = 1128
        self.filters = 16
        self.data_dir = 'data_cobot2'
C = Config()

def data_generator(T,C,le):
    X_0 = []
    X_1 = []
    Y = []
    for i in range(len(T['pose'])):
        p = np.copy(T['pose'][i])
        p = zoom(p,target_l=C.frame_l,joints_num=C.joint_n,joints_dim=C.joint_d)

        label = np.zeros(C.clc_num)
        label[le.transform(T['label'])[i]] = 1

        M = get_CG(p,C)

        X_0.append(M)
        X_1.append(p)
        Y.append(label)

    X_0 = np.stack(X_0)
    X_1 = np.stack(X_1)
    Y = np.stack(Y)
    return X_0,X_1,Y


# ----------------------------
# 1. Pose Difference Functions with Explicit Output Shapes
# ----------------------------

import tensorflow as tf

# Decorate your custom functions so that they are registered and can be found during deserialization.

@tf.keras.utils.register_keras_serializable()
def poses_diff(x):
    """
    Compute differences between consecutive frames in x and then use
    tf.image.resize to “restore” the original frame count.

    Assumes x has shape: (batch, frames, joints, dims)
    """
    orig_frames = tf.shape(x)[1]  # e.g., 32
    joint_n = tf.shape(x)[2]      # e.g., 22

    # Compute differences along the frame axis.
    diff = x[:, 1:, ...] - x[:, :-1, ...]

    # Resize the differences back to have the original number of frames.
    resized = tf.image.resize(diff, size=[orig_frames, joint_n])
    return resized

@tf.keras.utils.register_keras_serializable()
def poses_diff_func(x):
    # Simply a wrapper around poses_diff.
    return poses_diff(x)

def poses_diff_output_shape(input_shape):
    # Assuming the input shape is (batch, frames, joints, dims) and we intend to recover it.
    return input_shape


@tf.keras.utils.register_keras_serializable()
def sample_every_2(x):
    """
    Sample every 2nd frame from the input tensor.
    """
    return x[:, ::2, ...]

def sample_every_2_output_shape(input_shape):
    # Given input shape (batch, frames, joints, dims), return (batch, frames//2, joints, dims)
    batch = input_shape[0]
    frames = input_shape[1]
    joints = input_shape[2]
    dims = input_shape[3]
    return (batch, frames // 2, joints, dims)


def pose_motion(P, frame_l):
    """
    For a pose input P (shape: (batch, frame_l, joint_n, joint_d)),
    compute two kinds of differences:
      - Slow differences: computed on the full input.
      - Fast differences: computed on every 2nd frame.

    Then reshape the outputs so that the spatial dimensions are flattened.
    """
    # Slow branch using our registered function:
    P_diff_slow = Lambda(poses_diff_func, output_shape=poses_diff_output_shape)(P)
    P_diff_slow = Reshape((frame_l, -1))(P_diff_slow)

    # Fast branch:
    P_fast = Lambda(sample_every_2, output_shape=sample_every_2_output_shape)(P)
    P_diff_fast = Lambda(poses_diff_func, output_shape=poses_diff_output_shape)(P_fast)
    P_diff_fast = Reshape((frame_l // 2, -1))(P_diff_fast)

    return P_diff_slow, P_diff_fast

# ----------------------------
# 2. Convolutional and Dense Helper Blocks
# ----------------------------

def c1D(x, filters, kernel):
    x = Conv1D(filters, kernel_size=kernel, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def block(x, filters):
    x = c1D(x, filters, 3)
    x = c1D(x, filters, 3)
    return x

def d1D(x, filters):
    x = Dense(filters, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x

# ----------------------------
# 3. Build the Feature Model (FM)
# ----------------------------

def build_FM(frame_l=32, joint_n=22, joint_d=2, feat_d=231, filters=16):
    # Define inputs:
    M = Input(shape=(frame_l, feat_d))
    P = Input(shape=(frame_l, joint_n, joint_d))

    # Compute pose differences:
    diff_slow, diff_fast = pose_motion(P, frame_l)

    # Process the M input with a series of 1D convolutions.
    x = c1D(M, filters * 2, 1)
    x = SpatialDropout1D(0.1)(x)
    x = c1D(x, filters, 3)
    x = SpatialDropout1D(0.1)(x)
    x = c1D(x, filters, 1)
    x = MaxPooling1D(pool_size=2)(x)
    x = SpatialDropout1D(0.1)(x)

    # Process slow pose differences.
    x_d_slow = c1D(diff_slow, filters * 2, 1)
    x_d_slow = SpatialDropout1D(0.1)(x_d_slow)
    x_d_slow = c1D(x_d_slow, filters, 3)
    x_d_slow = SpatialDropout1D(0.1)(x_d_slow)
    x_d_slow = c1D(x_d_slow, filters, 1)
    x_d_slow = MaxPooling1D(pool_size=2)(x_d_slow)
    x_d_slow = SpatialDropout1D(0.1)(x_d_slow)

    # Process fast pose differences.
    x_d_fast = c1D(diff_fast, filters * 2, 1)
    x_d_fast = SpatialDropout1D(0.1)(x_d_fast)
    x_d_fast = c1D(x_d_fast, filters, 3)
    x_d_fast = SpatialDropout1D(0.1)(x_d_fast)
    x_d_fast = c1D(x_d_fast, filters, 1)
    x_d_fast = SpatialDropout1D(0.1)(x_d_fast)

    # Concatenate all three processed branches.
    x = concatenate([x, x_d_slow, x_d_fast])
    x = block(x, filters * 2)
    x = MaxPooling1D(pool_size=2)(x)
    x = SpatialDropout1D(0.1)(x)

    x = block(x, filters * 4)
    x = MaxPooling1D(pool_size=2)(x)
    x = SpatialDropout1D(0.1)(x)

    x = block(x, filters * 8)
    x = SpatialDropout1D(0.1)(x)

    return Model(inputs=[M, P], outputs=x)

# ----------------------------
# 4. Build the Final DD-Net Model
# ----------------------------

def build_DD_Net(C):
    """
    Assumes that C is an object or namespace with the following attributes:
      - C.frame_l: number of frames
      - C.feat_d: dimensionality of the M input features
      - C.joint_n: number of joints
      - C.joint_d: joint dimension (e.g., 2)
      - C.filters: base number of filters for the conv layers
      - C.clc_num: number of output classes
    """
    M = Input(name='M', shape=(C.frame_l, C.feat_d))
    P = Input(name='P', shape=(C.frame_l, C.joint_n, C.joint_d))

    FM = build_FM(C.frame_l, C.joint_n, C.joint_d, C.feat_d, C.filters)
    x = FM([M, P])

    x = GlobalMaxPooling1D()(x)
    x = d1D(x, 128)
    x = Dropout(0.5)(x)
    x = d1D(x, 128)
    x = Dropout(0.5)(x)
    x = Dense(C.clc_num, activation='softmax')(x)

    model = Model(inputs=[M, P], outputs=x)
    return model


DD_Net = build_DD_Net(C)
DD_Net.summary()


Train = pickle.load(open(C.data_dir+"/train_new_2.pickle", "rb"))
Test = pickle.load(open(C.data_dir+"/test_new_2.pickle", "rb"))

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(Train['label'])


label_percent = 0.8  # vd: chỉ dùng ..% mỗi class để train

def sample_by_label_percent(data_dict, label_percent, le):
    labels_encoded = le.transform(data_dict['label'])
    class_idx = {}
    for i, lbl in enumerate(labels_encoded):
        class_idx.setdefault(lbl, []).append(i)
    
    final_choice = []
    for lbl, idx_list in class_idx.items():
        n_sample = max(1, round(len(idx_list) * label_percent))  # ít nhất 1
        final_choice += random.sample(idx_list, n_sample)
    
    final_choice.sort()
    
    sampled_data = {
        'pose': [data_dict['pose'][i] for i in final_choice],
        'label': [data_dict['label'][i] for i in final_choice]
    }
    return sampled_data


Train_sampled = sample_by_label_percent(Train, label_percent, le)


X_0,X_1,Y = data_generator(Train,C,le)
X_test_0,X_test_1,Y_test = data_generator(Test,C,le)


import tensorflow as tf

# Set GPU as the only device

import os
# IMPORTANT: Set this before importing TensorFlow!
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
import tensorflow as tf

# Disable XLA JIT and force eager execution (for debugging purposes)
tf.config.optimizer.set_jit(False)
tf.config.run_functions_eagerly(True)  # Remove this line once debugging is complete

# Enable soft device placement so that if a GPU kernel fails, TF can fallback to CPU
tf.config.set_soft_device_placement(True)

# ------------------------------
# (Optional) Adjust BatchNormalization layers in your model
# If your model uses BatchNormalization, sometimes a very small epsilon can cause issues.
# Here we set the epsilon to 1e-3 for all BN layers in DD_Net.
for layer in DD_Net.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        print(f"Updating {layer.name}: epsilon {layer.epsilon} -> 1e-3")
        layer.epsilon = 1e-3
# ------------------------------
# Define the learning rate
lr = 0.01 #for training

# Compile your model with the desired loss, optimizer, and metrics.
DD_Net.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
    metrics=['accuracy']
)

# Create the ReduceLROnPlateau callback
lrScheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss',
    factor=0.1, #=> giảm 90%
    patience=10, #sau mỗi 10 epoch
    cooldown=5,
    min_lr=1e-5  # or use min_lr=5e-6 if preferred
)

checkpoint = ModelCheckpoint(
    'DD_Net_cobo_v1_100%_best.keras',
    monitor='loss',
    verbose=1,
    save_best_only=True,
    mode='min'
)

# Train the model on GPU:0
with tf.device("/GPU:0"):
    start_time = time.time()
    history = DD_Net.fit(
        [X_0, X_1],
        Y,
        batch_size=32,   # Adjust as needed
        epochs=400,      # Adjust as needed
        verbose=2,
        shuffle=True,
        callbacks=[lrScheduler, checkpoint],
    )
    print("--- Training completed in %s seconds ---" % (time.time() - start_time))
    model_path = 'DD_Net_cobo_v1_100%.keras'
    DD_Net.save(model_path)
