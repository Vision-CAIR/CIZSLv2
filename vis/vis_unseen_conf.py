# Visualize the classification confident of unseen classes w/ w/o discriminative loss
import numpy as np
import matplotlib.pyplot as plt
import os

# for without loss
# # input_file_path = 'test_disc_loss_constrains_100.txt'
# input_file_path = 'test_disc_noloss_constrains_all_cub1_easy1.txt'
# # output_file_path = 'out_disc_loss_constrains_100.txt'
# output_file_path = 'wo-constrains-noloss-cub-easy.txt'
# # processes_path = 'process_100.txt'
# processes_path = 'noloss-all.txt'

# for with loss
input_file_path = 'test_disc_loss_constrains_all.txt'
output_file_path = 'testtest_disc_loss_constrains_all_initial.txt'
processes_path = 'testtest_disc_loss_constrains_all_initial.txt'

batch_size = 1000


def file_preprocessing(input_file_path, output_file_path):
    with open(input_file_path, 'r') as in_file, open(output_file_path, 'a+') as out_file:
        out_length = 0
        for line in in_file:
            line = line.split()
            # print(len(line))
            for word in line:
                pure_word = word.replace('[', '').replace(']', ' ')
                # print(pure_word)
                if out_length % batch_size == 0 and out_length:
                    out_length = 0
                    out_file.write('\n')
                out_file.write(pure_word+' ')
                out_length += 1


def analysis(output_file_path):
    with open(output_file_path, 'r') as target_file, open(processes_path, 'a+') as pro_file:
        cal = 0
        line_mean = 0
        for line in target_file:
            line = line.split()
            # print('length is ', len(line)) # 1000 (batch_size)
            # if cal < 3:
            #     print(line, len(line))
            #     cal += 1
            # if len(line) == batch_size or len(line) == (batch_size+1):
            for word in line:
                line_mean += float(word)
            line_mean /= batch_size
            pro_file.write(str(line_mean)+'\n')


if __name__ == "__main__":
    file_preprocessing(input_file_path, output_file_path)
    analysis(output_file_path)







