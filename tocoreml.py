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

net = TransformerNet()
net.load_state_dict(torch.load(f"starry_night_10000.pth", map_location=torch.device('cpu')))
net.eval()

x = torch.rand(1, 3, 512, 512)
traced_model = torch.jit.trace(net, x)

model = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="input_image", shape=x.shape)]
)

model.save("starry_night.mlmodel")