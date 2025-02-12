import torch.nn as nn
import torchvision.models as models  


def res34(pretrained = True, fine_tune = True, num_classes = 2):
    """
    num_classes: number of classifications. '+' or '-'.
    """
    if pretrained:
        print('[INFO]: Loading pre-trained weights...')  
    elif not pretrained:
        print('[INFO]: Not loading pre-trained weights...')  
    model = models.resnet34(pretrained=pretrained)
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for param in model.parameters():
            param.requires_grad = False
        # paragraghs below are selectable "True" for better performance:
        # defreeze the last block.
        for param in model.layer4.parameters():
            param.requires_grad = True
        # defreeze the 3rd block.
        for param in model.layer3.parameters():
            param.requires_grad = True
        # defreeze the 2rd block.    
        for param in model.layer2.parameters():
            param.requires_grad = False
        # defreeze the 1st block.
        for param in model.layer1.parameters():
            param.requires_grad = False
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    # change the final classification head, trainable.
    # model.fc = nn.Linear(512, num_classes)
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2))
    return model

def kan(pretrained = True, fine_tune = True, num_classes = 2):
    """
    num_classes: number of classifications. '+' or '-'.
    """
    if pretrained:
        print('[INFO]: Loading pre-trained weights...')  
    elif not pretrained:
        print('[INFO]: Not loading pre-trained weights...')
    model = kan(pretrained=pretrained)

    return model