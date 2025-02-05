from ultralytics import YOLO
import wandb
wandb.init(project='yolo-train')
# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model with MPS
results = model.train(data="/home/kirti/Training/datasets/ptz_seg_polygon_Aug/ptz_seg_polygon-copy1-1/data.yaml", epochs=300, imgsz=640, plots=True, save_period=10, lr0=0.00)