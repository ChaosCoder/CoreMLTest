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
from image import convert_multiarray_output_to_image
import copy

model = ct.models.MLModel('starry_night.mlmodel')

spec = model.get_spec()
spec.description.output[0].type.multiArrayType.shape.extend([3, 512, 512])

red_b = -(0.485 * 255.0)
green_b = -(0.456 * 255.0)
blue_b = -(0.406 * 255.0)

red_scale = 1.0 / (0.229 * 255.0)
green_scale = 1.0 / (0.224 * 255.0)
blue_scale = 1.0 / (0.225 * 255.0)

args = dict(is_bgr=False, red_bias = red_b, green_bias = green_b, blue_bias = blue_b)
nn_spec = spec.neuralNetwork
layers = nn_spec.layers # this is a list of all the layers
layers_copy = copy.deepcopy(layers) # make a copy of the layers, these will be added back later
del nn_spec.layers[:] # delete all the layers

# add a scale layer now
# since mlmodel is in protobuf format, we can add proto messages directly
# To look at more examples on how to add other layers: see "builder.py" file in coremltools repo
scale_layer = nn_spec.layers.add()
scale_layer.name = 'scale_layer'
scale_layer.input.append('input_image')
scale_layer.output.append('0_scaled')

params = scale_layer.scale
params.scale.floatValue.extend([red_scale, green_scale, blue_scale]) # scale values for RGB
params.shapeScale.extend([3,1,1]) # shape of the scale vector 

# now add back the rest of the layers (which happens to be just one in this case: the crop layer)
nn_spec.layers.extend(layers_copy)

# need to also change the input of the crop layer to match the output of the scale layer
nn_spec.layers[1].input[0] = '0_scaled'

# find the current output layer and save it for later reference
last_layer = nn_spec.layers[-1]
 
# add the post-processing layer
new_layer = nn_spec.layers.add()
new_layer.name = 'output_image'
 
# Configure it as an activation layer
new_layer.activation.linear.alpha = 255
new_layer.activation.linear.beta = 0
 
# Use the original model's output as input to this layer
new_layer.input.append(last_layer.output[0])
 
# Name the output for later reference when saving the model
new_layer.output.append('output_image')
 
# Find the original model's output description
output_description = next(x for x in spec.description.output if x.name==last_layer.output[0])
output_description.name = new_layer.name

convert_multiarray_output_to_image(spec, 'output_image', is_bgr=False)

updated_model = ct.models.MLModel(spec)

updated_model.user_defined_metadata["com.apple.coreml.model.preview.type"] = "image"
updated_model.input_description['input_image'] = 'Input Image'
updated_model.output_description[output_description.name] = 'Predicted Image'

updated_model.save('starry_night_with_output.mlmodel')