# import os
# from ultralytics import YOLO
# from ultralytics.engine.results import Results

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--device', type=str, default='0')
# args = parser.parse_args()

# os.environ["ACCELERATE_TORCH_DEVICE"] = 'cpu' if args.device == 'cpu' else 'dlc'
# device = 'cpu' if args.device == 'cpu' else int(args.device)
# home_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

# model = YOLO(os.path.join(home_dir, 'ci', 'yolov8n.pt'), verbose=True)
# image = os.path.join(home_dir, 'ultralytics', 'assets', 'bus.jpg')
# # Use the model
# # results = model.train(data='coco8.yaml', epochs=3)  # train the model

# # results = model.val(data='coco.yaml', device=device)  # evaluate model performance on the validation set
# results: Results = model(image)[0]  # predict on an image
# results.save(os.path.join(home_dir, 'ci', 'out.jpg'))
from ultralytics import YOLO
import os
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0')
args = parser.parse_args()
os.environ["ACCELERATE_TORCH_DEVICE"] = 'cpu' if args.device == 'cpu' else 'dlc'
device = 'cpu' if args.device == 'cpu' else int(args.device)

model_seg = YOLO("weights/yolov8n-seg_1.pt")
model_detect = YOLO("weights/yolov8n.pt")
model_classify = YOLO("weights/yolo11n-cls.pt")
model_pose = YOLO("weights/yolov8m-pose.pt")
model_obb = YOLO("weights/yolo11n-obb.pt")

def testSeg(model):
    results = model.predict("inputs/elephants_seg.jpg")
    for result in results:
        boxes = result.boxes    
        result.save("outputs/result_elephants_seg.jpg") 

def testDetect(model):
    results = model.predict("inputs/elephants_det.jpg")
    for result in results:
        boxes = result.boxes    
        result.save("outputs/result_elephants_det.jpg")
        
def testClassify(model):
    results = model.predict("inputs/bus_classify.jpg")
    for result in results:
        result.save("outputs/result_bus_classify.jpg")

def testPose(model):
    results = model(source="inputs/running.mp4", show = False, conf = 0.3, save = True, project = "outputs", name = "running_Pose.avi")

def testOBB(model):
    results = model(source="inputs/water.mp4", show = False, conf = 0.3, save = True, project = "outputs", name = "water_OBB.avi")
    
def track():
    results_seg = model_seg(source="inputs/running.mp4", show = False, conf = 0.3, save = True, project = "outputs", name = "running_seg.avi")
    results_det = model_det(source="inputs/running.mp4", show = False, conf = 0.3, save = True, project = "outputs", name = "running_det.avi")
    result_pose = model_pose(source="inputs/running.mp4", show = False, conf = 0.3, save = True, project = "outputs", name = "running_pose.avi")
    
#testSeg(model_seg)
testDetect(model_detect)
#testClassify(model_classify)
#testPose(model_pose)
#testOBB(model_obb)
#track()
