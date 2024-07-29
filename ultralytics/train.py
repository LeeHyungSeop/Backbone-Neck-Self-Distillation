from ultralytics import YOLO

# Load a model
model = YOLO("yolov10n.yaml")  # build a new model from YAML
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="coco.yaml", epochs=100, imgsz=640, device=[0], pretrained=False, save_txt = True, project="runs/detect", name="yolov10n")
'''
python train.py 2>&1 | tee ./logs/test.txt
'''