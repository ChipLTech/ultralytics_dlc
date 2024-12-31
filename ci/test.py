import os
from ultralytics import YOLO
from ultralytics.engine.results import Results

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0')
args = parser.parse_args()

os.environ["ACCELERATE_TORCH_DEVICE"] = 'cpu' if args.device == 'cpu' else 'dlc'
device = 'cpu' if args.device == 'cpu' else int(args.device)

model = YOLO('yolov8x.pt', verbose=True)  # load a pretrained model (recommended for training)

# Use the model
# results = model.train(data='coco8.yaml', epochs=3)  # train the model
# import ipdb; ipdb.set_trace()
results = model.val(data='coco.yaml', device=device)  # evaluate model performance on the validation set
# results: Results = model('/workspace/test/dlc/datasets/coco/images/val2017/000000278705.jpg')[0]  # predict on an image
# results.save("out.jpg")
