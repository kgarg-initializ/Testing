# from ray import tune

# from ultralytics import YOLO
# import wandb

# wandb.init(project='yolo11_tuning')
# # Define a YOLO model
# model = YOLO("yolo11s.pt")

# # Run Ray Tune on the model
# result_grid = model.tune(
#     data="/home/jovyan/Training/datasets/ptz_seg_polygon_Aug/ptz_seg_polygon-copy1-1/data.yaml",
#     space={"lr0": tune.uniform(1e-5, 1e-1)},
#     iterations=100,
#     epochs=300,
#     use_ray=True,
#     plots=False,
#     save=False,
#     val=False,
# )

# from ray import tune

from ultralytics import YOLO
import wandb
from ray import tune
wandb.login(key="ef999c477c85e5d7f73f966fdc49c047c23a5987")
search_space={"lr0": tune.uniform(1e-5, 1e-1), "weight_decay": tune.uniform(0.0, 0.001)}
wandb.init(project='yolo11_tuning')
# Define a YOLO model
model = YOLO("yolo11x-seg.pt")

# Run Ray Tune on the model
result_grid = model.tune(
    data="/Testing/ptz_seg_polygon_Aug/ptz_seg_polygon-copy1-1/data.yaml",
    space=search_space,
    iterations=100,
    epochs=300,
    use_ray=True,
    plots=False,
    save=False,
    val=False,
)
