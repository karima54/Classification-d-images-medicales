import json
from commons import get_model, transform_image
from PIL import Image
import torch
import onnx
import os
import onnxruntime

"""
Load the trained model and dictionary contains images classes
"""
model_path = os.path.join('models', 'MedNet.onnx')
# model_path = os.path.join('models', 'MedNet_v2.onnx')
ort_session = onnxruntime.InferenceSession(model_path)
imagenet_class_index = json.load(open('imagenet_class_index.json'))

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def get_prediction(image_bytes):
    """
    Get imported image as Bytes
    scale the image (normalize, centralize), pass it to the trained model to get prediction
    Return:
    image prediction
    """
    try:
        img_y = transform_image(image_bytes)
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
        ort_outs = ort_session.run(None, ort_inputs)
        outputs = ort_outs[0]
    except Exception:
        return 404, 'error'
    print(imagenet_class_index.get(str(outputs.argmax())), outputs.argmax())
    return imagenet_class_index.get(str(outputs.argmax())), outputs.argmax()