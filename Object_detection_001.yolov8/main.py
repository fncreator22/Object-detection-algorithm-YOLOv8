from ultralytics import YOLO

model = YOLO("yolov8n.yaml")
result = model.train(data=r"C:\Users\Sagar\Downloads\object_detection_001\Object_detection_001.v1i.yolov8\data.yaml", epochs=5, imgsz=640, deterministic=True)
