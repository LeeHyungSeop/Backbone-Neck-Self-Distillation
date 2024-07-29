from ultralytics import YOLO

# Load a model
model = YOLO("yolov5m.yaml")  # build a new model from YAML
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="coco.yaml", epochs=100, imgsz=640, device=[0], pretrained=False, save_txt = True, project="runs/detect", name="yolov10n")
'''
python train.py 2>&1 | tee ./logs/test.txt

YOLOv10n summary: 385 layers, 2,775,520 parameters, 2,775,504 gradients, 8.7 GFLOPs
YOLOv10m summary: 498 layers, 16,576,768 parameters, 16,576,752 gradients, 64.5 GFLOPs

YOLOv5s summary: 262 layers, 9,153,152 parameters, 9,153,136 gradients, 24.2 GFLOPs
YOLOv5m summary: 339 layers, 25,111,456 parameters, 25,111,440 gradients, 64.6 GFLOPs
'''