from flask import Flask, request
from ultralytics import YOLO
import os

app = Flask(__name__)
model = YOLO("yolov8n.yaml")

@app.route('/train', methods=['POST'])
def train_model():
    image_file = request.files['image']
    temp_image_path = 'temp_image.jpg'
    image_file.save(temp_image_path)
    result = model.train(data=temp_image_path, epochs=1, imgsz=640, deterministic=True)
    os.remove(temp_image_path)  # Delete the temporary image file
    return 'Model trained successfully.'

if __name__ == '__main__':
    app.run(debug=True)
