"""
Run once ONLY when create a dataset.
"""

import os
import random
import shutil

def create_sample_folders(base_dir):

    high_dir = os.path.join(base_dir, 'high')
    low_dir = os.path.join(base_dir, 'low')
    a_dir = os.path.join(base_dir, 'a')
    b_dir = os.path.join(base_dir, 'b')
    c_dir = os.path.join(base_dir, 'c')
    d_dir = os.path.join(base_dir, 'd')

    os.makedirs(a_dir, exist_ok=True)
    os.makedirs(b_dir, exist_ok=True)
    os.makedirs(c_dir, exist_ok=True)
    os.makedirs(d_dir, exist_ok=True)

    high_files = os.listdir(high_dir)
    low_files = os.listdir(low_dir)

    high_sample = random.sample(high_files, 5298)
    high_rest = set(high_files) - set(high_sample)

    low_sample = random.sample(low_files, 3086)
    low_rest = set(low_files) - set(low_sample)

    for file in high_sample:
        shutil.move(os.path.join(high_dir, file), os.path.join(a_dir, file))
    for file in high_rest:
        shutil.move(os.path.join(high_dir, file), os.path.join(c_dir, file))
    for file in low_sample:
        shutil.move(os.path.join(low_dir, file), os.path.join(b_dir, file))
    for file in low_rest:
        shutil.move(os.path.join(low_dir, file), os.path.join(d_dir, file))

base = 'all_JPG'
create_sample_folders(base)
