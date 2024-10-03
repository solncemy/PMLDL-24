from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow import keras
import os
import joblib


app = FastAPI()
model = joblib.load('/app/models/mnist_model.pkl')

class InputData(BaseModel):
    image: list  
    
# https://www.kaggle.com/code/raibektussupbekov/downsampling-algorithm
def downsample_image(img:np.array, ratio: float) -> np.array:
    
    """
    Downsamples or scales down an image.
    
    Keyword arguments:
    img -- the image data, 
           Numpy ndarray has shape either
           * (H, W) for grayscale images
           or
           * (H, W, 3) for RGB images
           or
           * (H, W, 4) for RGBA images
    ratio -- the extent to scale down the image
    
    Return:
    the downsampled copy of the image data,
    Numpy ndarray has the same shape and dtype as the input
    """
    
    height, width = img.shape[:2]

    height_new = height // ratio
    width_new = width // ratio
    
    height_scale = height / height_new
    width_scale = width / width_new
    
    # RGB(A)
    if len(img.shape) > 2:
        img_new = np.zeros((height_new, width_new, img.shape[2]), dtype=img.dtype)
        for channel in range(img.shape[2]):
            for x_new in range(height_new):
                for y_new in range(width_new):
                    img_new[x_new, y_new, channel] = img[round(x_new*height_scale), round(y_new*width_scale), channel]
    # Grayscale
    else:
        img_new = np.zeros((height_new, width_new), dtype=img.dtype)
        for x_new in range(height_new):
            for y_new in range(width_new):
                img_new[x_new, y_new] = img[round(x_new*height_scale), round(y_new*width_scale)]
                
    return img_new

@app.post("/predict")
def predict(data: InputData):
    image = np.array(data.image).reshape(280, 280) 
    image = downsample_image(image, 10)
    image = image.reshape(1, 28, 28, 1) / 255
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return {"prediction": int(predicted_class)}
