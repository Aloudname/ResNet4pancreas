import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# parameters.
train_batch_size = 64
test_batch_size = 16
train_dir = 'input\\train'
test_dir = 'input\\test'

# pre processing.
train_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 5)),
    # transforms.RandomRotation(degrees=(30, 70)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ])

valid_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ])

# initial entire and test datasets. Used in train.py
dataset_train = datasets.ImageFolder(train_dir, transform=train_transform)
dataset_valid = datasets.ImageFolder(test_dir, transform=valid_transform)

train_size, valid_size = len(dataset_train), len(dataset_valid)
print(f"Classes: {dataset_train.classes}\n")
print(f"Total number of images: {train_size + valid_size}\n")

# training and validation sets.
indices = torch.randperm(len(dataset_train)).tolist()

print(f"Total training images: {len(dataset_train)}")
print(f"Total valid_images: {len(dataset_valid)}")

# training and validation data loaders. Used in train.py
train_loader = DataLoader(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(dataset_valid, batch_size=test_batch_size, shuffle=True, num_workers=0)


if __name__ == "__main__":
    print('torch version: ' + torch.__version__)
    print('torchvision version: ' + torchvision.__version__)
    print('if cuda available: ' + str(torch.cuda.is_available()))
    print(f'number of batch: {train_size / train_batch_size}')