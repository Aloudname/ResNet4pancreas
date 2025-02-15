import cv2
import torch
import argparse
import torchvision.transforms as transforms

from model import res34


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input',
    default='E:\\yixian\\input\\test\\high\\caokeyun-A_slice_226.jpg',
    help='path to the input image')
args = vars(parser.parse_args())
device = 'cpu'
labels = ['high', 'low']

# initialization and trained weights loading.
model = res34(
    pretrained = False, fine_tune = False, num_classes= 2
    ).to(device)
print('[INFO]: Loading custom-trained weights...')
checkpoint = torch.load('outputs/model.pth', map_location = device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# define preprocess transforms.
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# read and preprocess the image.
image = cv2.imread(args['input'])  
# get the ground-truth(gt) class.  
gt_class = args['input'].split('/')[-1].split('.')[0]  
orig_image = image.copy()

# convert to RGB.
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
image = transform(image)

# add batch dimension.
image = torch.unsqueeze(image, 0)

with torch.no_grad():  
    outputs = model(image.to(device))
    output_label = torch.topk(outputs, 1)
    pred_class = labels[int(output_label.indices)]
cv2.putText(orig_image,  
    f"GT: {gt_class}",  
    (10, 25),  
    cv2.FONT_HERSHEY_SIMPLEX,  
    1, (0, 255, 0), 2, cv2.LINE_AA  
)  
cv2.putText(orig_image,  
    f"Pred: {pred_class}",  
    (10, 55),  
    cv2.FONT_HERSHEY_SIMPLEX,  
    1, (0, 0, 255), 2, cv2.LINE_AA  
)  

print(f"GT: {gt_class}, pred:{pred_class}")
cv2.imshow('result', orig_image)
cv2.waitkey(0)
cv2.imwrite(f"outputs/{gt_class}.png", orig_image)