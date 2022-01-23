from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import uvicorn
import json

app = FastAPI()

model_path = "../model/tf_keras_cifar"
model = tf.keras.models.load_model(model_path)

with open("../model/labels.json", "r") as f:
    label = json.load(f)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = tf.io.decode_image(contents, channels=3)
    image = tf.image.resize(image, (32, 32))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    print(np.min(image), np.max(image))
    result = model.predict(image)
    return {"result: ": label[str(np.argmax(result[0]))]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
