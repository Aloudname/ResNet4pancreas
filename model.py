import torch
import torch.nn as nn
import torch.nn.functional as F
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
            param.requires_grad = True
        # defreeze the 1st block.
        for param in model.layer1.parameters():
            param.requires_grad = True

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


class KANLayer(nn.Module):
    """A single KAN layer derived from nn.Module."""
    def __init__(self, input_dim, output_dim, num_basis=5):
        """
        num_basis : Number of basis functions, reflects the complexity of a layer.
        weights: Learnable weights of basis functions.
        basis_functions: Pattern of the basis function. Linear defaultly.
        """
        super().__init__()
        self.num_basis = num_basis
        self.weights = nn.Parameter(torch.randn(output_dim, input_dim, num_basis))
        self.basis_functions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 16),
                nn.SiLU(),
                nn.Linear(16, 1)
            ) for _ in range(num_basis)
        ])

    def forward(self, x):
        # x shape: [batch, input_dim]
        batch_size, input_dim = x.shape
        outputs = []
        for out_dim in range(self.weights.shape[0]):
            out = 0.0
            for in_dim in range(input_dim):
                basis_input = x[:, in_dim].unsqueeze(1)  # [batch, 1]
                basis_output = self.basis_functions[in_dim](basis_input)  # [batch, 1]
                weight = self.weights[out_dim, in_dim]  # [num_basis]
                out += torch.sum(weight * basis_output, dim=-1)  # [batch]
            outputs.append(out)
        return torch.stack(outputs, dim = 1)  # [batch, output_dim]

class KAN(nn.Module):
    """KAN module for 512*512 images classification."""
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),  # 256x256
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2),  # 128x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2),  # 64x64
            nn.Flatten()  # output dim: 128*64*64 = 524288
            )
        
        self.kan_layers = nn.Sequential(
            KANLayer(input_dim=524288, output_dim=256),
            nn.SiLU(),
            KANLayer(input_dim=256, output_dim=64),
            nn.SiLU(),
            KANLayer(input_dim=64, output_dim=16))
        
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x):
        # input x: [batch, 3, 512, 512]
        features = self.feature_extractor(x)  # [batch, 524288]
        kan_out = self.kan_layers(features)   # [batch, 16]
        logits = self.classifier(kan_out)     # [batch, 2]
        return logits

if __name__ == "__main__":
    model = KAN(in_channels=3, num_classes=2)
    input_tensor = torch.randn(4, 3, 512, 512)  # 4 images of 512x512 RGB.
    output = model(input_tensor)
    print("Output shape:", output.shape)  # [4, 2].