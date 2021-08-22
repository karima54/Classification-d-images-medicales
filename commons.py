
import os
import io
import torchvision.transforms as transforms
import torchvision as tv
import torch
from PIL import Image
import onnx
import json

def get_model():
    # Preprocessing: load the ONNX model
    model_path = os.path.join('models', 'MedNet.onnx')
    # model_path = os.path.join('models', 'MedNet_v2.onnx')
    model = onnx.load(model_path)
    # Check the model
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        print('The model is invalid: %s' % e)
    else:
        print('The model is valid!')
    return model

# Pass a PIL image, return a tensor
def scaleImage(x):
    """
    Normalize & center the image pixels
    """
    toTensor = tv.transforms.ToTensor()
    y = toTensor(x)
    if(y.min() < y.max()):
        y = (y - y.min()) / (y.max() - y.min())
    z = y - y.mean()
    return z

def transform_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img_y = scaleImage(img)
    img_y.unsqueeze_(0)
    return img_y

def format_class_name(class_name):
    class_name = class_name.title()
    return class_name

def create_predictions_folders():
    prediction_dir = "./predictions"
    os.makedirs(prediction_dir, exist_ok=True)
    imagenet_class_index = json.load(open('imagenet_class_index.json'))
    for index, classe in imagenet_class_index.items():
        # print(classe)
        os.makedirs(os.path.join(prediction_dir, classe), exist_ok=True)
    return prediction_dir