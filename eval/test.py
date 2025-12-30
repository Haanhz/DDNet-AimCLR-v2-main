import numpy as np

# data = np.load("data_cobot_clr/xsub/val_position.npy", mmap_mode='r')
# print(data.shape)

# for frame in data:
#     print(frame.shape)

X = np.random.rand(60,48,3)
finite_mask = np.isfinite(X).all(axis=2)
print(finite_mask)
print(finite_mask.shape)
# Then, detect frames where ALL coordinates are exactly zero (likely gaps)
all_zero_mask = (X == 0).all(axis=2)
print(all_zero_mask)
print(all_zero_mask.shape)
# A joint is valid if it's finite AND not all-zero
mask = finite_mask & ~all_zero_mask
print(mask)
print(mask.shape)