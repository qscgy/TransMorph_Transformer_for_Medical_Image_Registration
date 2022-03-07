import glob
import os
import pickle
import random
from os.path import isdir, isfile, join

base_dir = '/playpen/meshes-better'
val_split = 0.1

numbers = [format(i, '03d') for i in range(1,101)]
numbers.remove('061')
numbers.remove('078')
numbers.remove('079')
files = [[f'{base_dir}/sim_{n}.obj', f'{base_dir}/sim_{n}_deformed.obj'] for n in numbers]

random.shuffle(files)
if len(files) % 2 ==1:
    files = files[:-1]

val_list = files[:int(len(files)*val_split)]
train_list = files[int(len(files)*val_split):]
print(len(files))
print(val_list)

with open('affine_splits.pkl', 'wb') as f:
    pickle.dump({'val_list':val_list, 'train_list':train_list}, f)