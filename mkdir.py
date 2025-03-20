import os
import random
import shutil

def split_dataset(root_dir):
    """
    Only used when split the dataset.
    """
    for subdir_name in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir_name)
        
        # skip the file in other types.
        if not os.path.isdir(subdir_path) or subdir_name.startswith('.'):
            continue

        png_files = []
        for file_name in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith('.png'):
                png_files.append(file_name)
        if not png_files:
            continue

        random.shuffle(png_files)
        

        split_idx = int(len(png_files) * 64 / (64 + 16))
        train_files = png_files[:split_idx]
        test_files = png_files[split_idx:]
        
        train_dir = os.path.join(subdir_path, 'train')
        test_dir = os.path.join(subdir_path, 'test')
        for folder in [train_dir, test_dir]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for file_name in train_files:
            src = os.path.join(subdir_path, file_name)
            dst = os.path.join(train_dir, file_name)
            shutil.move(src, dst)

        for file_name in test_files:
            src = os.path.join(subdir_path, file_name)
            dst = os.path.join(test_dir, file_name)
            shutil.move(src, dst)

if __name__ == '__main__':
    split_dataset("C:/Users/34739/Desktop/COVID-19_Radiography_Dataset")