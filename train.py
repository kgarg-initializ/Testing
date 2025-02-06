from ultralytics import YOLO
import wandb
wandb.login(key="ef999c477c85e5d7f73f966fdc49c047c23a5987")
wandb.init(project='yolo-train')
# Load a model
model = YOLO("yolo11x-seg.pt")  # load a pretrained model (recommended for training)

# Train the model with MPS
results = model.train(data="/ptz_seg_polygon_Aug/ptz_seg_polygon-copy1-1/data.yaml", epochs=300, imgsz=640, plots=True, save_period=10, weight_decay=0.00075))
