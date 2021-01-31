import torch
import torchvision
from torchvision import datasets, models, transforms
from models import TransformerNet,VGG16
import coremltools as ct
import urllib
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import numpy as np

def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return tensors

net = TransformerNet()

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def style(style, filename):
    input_image = Image.open(f"{filename}.jpg")
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    x = input_batch

    net.load_state_dict(torch.load(f"{style}_10000.pth", map_location=torch.device('cpu')))
    net.eval()

    y = net(x)

    save_image(denormalize(y), f"{filename}_{style}.jpg")

style("mosaic", "image2")
style("starry_night", "image2")
style("cuphead", "image2")

# label_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
# class_labels = urllib.request.urlopen(label_url).read().decode("utf-8").splitlines()

# class_labels = class_labels[1:] # remove the first class which is background
# assert len(class_labels) == 1000

# print(y.shape)
# print(torch.argmax(y))
# print(torch.max(y))

# print(class_labels[torch.argmax(y)])