import numpy as np

target_pos_x = 15
target_pos_y = -15

tar_x_pos = np.maximum(0, target_pos_x)  # positv x or 0
tar_x_neg = np.minimum(0, target_pos_x)  # negativ x or 0
tar_y_pos = np.maximum(0, target_pos_y)  # positv y or 0
tar_y_neg = np.minimum(0, target_pos_y)  # negativ y or 0

print(tar_x_pos)
print(tar_x_neg)
print(tar_y_pos)
print(tar_y_neg)