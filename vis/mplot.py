import matplotlib.pyplot as plt
import os
import numpy as np

# from scipy.interpolate import spline

# mfile_path = 'process_100.txt'
# save_image_path = 'test.png'
mfile_path = 'constrains_loss_cub1_easy1.txt'
save_image_path = 'noloss_cub_easy2.png'


with open(mfile_path, 'r') as mfile:
    y_axis = []
    for line in mfile:
        y_axis.append(float(line))
    # print(len(y_axis))
    x_axis = np.arange(len(y_axis))

    plt.title('Highest classification confidence with all batches')
    plt.plot(x_axis, y_axis, color='blue', label='w/o discriminative loss', linestyle='-')
    plt.legend()
    plt.savefig(save_image_path)
    plt.show()

