import os
from ultralytics import YOLO
from ultralytics.engine.results import Results

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0')
args = parser.parse_args()

os.environ["ACCELERATE_TORCH_DEVICE"] = 'cpu' if args.device == 'cpu' else 'dlc'
device = 'cpu' if args.device == 'cpu' else int(args.device)
home_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

model = YOLO(os.path.join(home_dir, 'ci', 'yolov8n.pt'), verbose=True)
image = os.path.join(home_dir, 'ultralytics', 'assets', 'bus.jpg')
# Use the model
# results = model.train(data='coco8.yaml', epochs=3)  # train the model

# results = model.val(data='coco.yaml', device=device)  # evaluate model performance on the validation set
results: Results = model(image)[0]  # predict on an image
results.save(os.path.join(home_dir, 'ci', 'out.jpg'))
