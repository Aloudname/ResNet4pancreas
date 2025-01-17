import torch  
import argparse
import torch.nn as nn
from tqdm.auto import tqdm
import torch.optim as optim

from model import build_model  
from utils import save_model, save_plots  
from datasets import train_loader, valid_loader, dataset_valid


# construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=30,  
help='number of epochs to train network for')
args = vars(parser.parse_args())
epochs = args['epochs']

lr = 0.001
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nComputation device: {device}")

# model.
model = build_model(
    pretrained = True, fine_tune = False,
    num_classes = len(dataset_valid.classes)
    ).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.\n")

def train(model, trainloader, optimizer, criterion):
    """
    4 basic objects required.
    model: model to train.
    trainloader: DataLoader for training data.
    optimizer: optimizer for training.
    criterion: loss criterion for training.
    """

    model.train()
    print('Training...')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for index, (data, targets) in tqdm (enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data, targets
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # forward.
        outputs = model(image)
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        
        # back propagation, update the optimizer parameters
        loss.backward()
        optimizer.step()

    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))

    return epoch_loss, epoch_acc

def validate(model, testloader, criterion, class_names):  
    """
    4 basic objects required.
    model: model trained(to be tested).
    testloader: DataLoader for test data.
    criterion: loss criterion for training.
    class_names: list of class names.
    """

    model.eval()  
    print('Validation:\n')  
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0

    # 2 lists required to keep track of class-wise accuracy.
    class_correct = list(0. for i in range(len(class_names)))  
    class_total = list(0. for i in range(len(class_names)))  
    
    with torch.no_grad():
        for index, (data, targets) in tqdm(enumerate(testloader), total = len(testloader)):  
            counter += 1
            image, labels = data, targets
            image = image.to(device)
            labels = labels.to(device)
            # forward.
            outputs = model(image)
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

            # calculate the accuracy for each class.
            # correct = (preds == labels).squeeze()
            correct = (preds == labels)
            print(f"Correct samples :{correct}\n")
            for i in range(len(preds)):
                label = labels[i]

                class_correct[label] += correct[i].item()
                class_total[label] += 1
            print(f"Total labels:{class_correct}\n")

    # for completed epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))

    # accuracy for each class per epoch.
    print('\n')
    for j in range(len(class_names)):
        print(f"Accuracy of class {class_names[j]}: {100*class_correct[j] / class_total[j]}")
    print('\n')

    return epoch_loss, epoch_acc


# lists to keep track of losses and accuracies  
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []

# training. 
for epoch in range(epochs):  
    print(f"[INFO]: Epoch {epoch+1} of {epochs}]")  
    train_epoch_loss, train_epoch_acc = train(model, train_loader,  
                                             optimizer, criterion)  
    valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,  
                                                 criterion, ['high', 'low'])

    train_loss.append(train_epoch_loss)  
    valid_loss.append(valid_epoch_loss)  
    train_acc.append(train_epoch_acc)  
    valid_acc.append(valid_epoch_acc)  

    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")  
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")  
    print("-" * 15)  
  
save_model(epochs, model, optimizer, criterion)
save_plots(train_acc, valid_acc, train_loss, valid_loss)  
print('TRAINING COMPLETE!')