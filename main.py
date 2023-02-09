from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from numpy import asarray
from io import BytesIO
import cv2
import numpy as np
import base64
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef
import os

app = FastAPI(title='Brain Tumor Segmentation', version='1.0',
              description='Unet Architecture model is used for prediction')

with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss}):
    model = tf.keras.models.load_model(os.path.join("/files", "model.h5"))

@app.post("image_input/")
def image_input(file: UploadFile):
    binary_data = file.file.read()
    file = BytesIO(binary_data)
    image = cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256))
    x = image/255.0
    x = np.expand_dims(x, axis=0)
    result = segment(x)
    ret, buffer = cv2.imencode('.jpg', result)
    return StreamingResponse(BytesIO(buffer), media_type="image/jpeg")

def segment(x):
        y_pred = model.predict(x, verbose=0)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred >= 0.5
        y_pred = y_pred.astype(np.int32)
        
        y_pred = np.expand_dims(y_pred, axis=-1)
        y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
        y_pred = y_pred * 255
        return y_pred
        