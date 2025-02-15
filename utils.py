import torch  
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

matplotlib.style.use('ggplot')  

def save_model(epochs, model, optimizer, criterion):  
    """
    Save the trained model to disk.  
    """

    torch.save(
        {  
        'epoch': epochs,  
        'model_state_dict': model.state_dict(),  
        'optimizer_state_dict': optimizer.state_dict(),  
        'loss': criterion,  
        }, 'outputs/model.pth')

def save_acc(train_acc, valid_acc, train_loss, valid_loss):  
    """
    Save the loss and accuracy plots to disk.  
    """

    # accuracy plots  
    plt.figure(figsize=(10, 7))  
    plt.plot(  
        train_acc, color='green', linestyle='-',  
        label='train accuracy'  
    )  
    plt.plot(  
        valid_acc, color='blue', linestyle='-',  
        label='validataion accuracy'  
    )  
    plt.xlabel('Epochs')  
    plt.ylabel('Accuracy')  
    plt.legend()  
    plt.savefig('outputs/accuracy.png')  
    # loss plots  
    plt.figure(figsize=(10, 7))  
    plt.plot(  
        train_loss, color='orange', linestyle='-',  
        label='train loss'  
    )  
    plt.plot(
        valid_loss, color = 'red', linestyle = '-',
        label = 'validation loss'
    )
    plt.xlabel("Epochs")
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/loss.png')

def save_matrix(true_labels, pred_labels, class_names):
    """
    Save the confusion matrix.
    """
    filename='outputs/confusion_matrix.png'
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=class_names, 
                yticklabels=class_names,
                cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix saved to {filename}")