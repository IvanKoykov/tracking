from models import *
from utils import utils
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import time
from sort_v2 import *
from PIL import Image

config_path = 'config/yolov3.cfg'
weights_path = 'config/yolov3.weights'
class_path = 'config/coco.names'
img_size = 416
conf_thres = 0.8
nms_thres = 0.4

# load model and put into eval mode
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.eval()

classes = utils.load_classes(class_path)
Tensor = torch.FloatTensor


def detect_image(img):
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2), 0), max(int((imw-imh)/2), 0),
                         max(int((imh-imw)/2), 0), max(int((imw-imh)/2), 0)),
                        (128, 128, 128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))

    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]


videopath = 'test_video.avi'

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
          (255, 0, 255), (128, 0, 0), (0, 128, 0),
          (0, 0, 128), (128, 0, 128), (128, 128, 0), (0, 128, 128)]

vid = cv2.VideoCapture(videopath)
mot_tracker = Sort() 

# cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Stream', (800,600))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret, frame = vid.read()
vw = frame.shape[1]
vh = frame.shape[0]
print ("Video size", vw,vh)
outvideo = cv2.VideoWriter(videopath.replace(".avi", "-det.avi"),fourcc,20.0,(vw,vh))
frames = 0
starttime = time.time()
way = {}
while True:
    ret, frame = vid.read()
    if not ret:
        break
    frames += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    if detections is not None:
        tracked_objects = mot_tracker.update(detections.cpu())

        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            color = colors[int(obj_id) % len(colors)]
            cls = classes[int(cls_pred)]
            cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
            cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+80, y1), color, -1)
            cx = int((x1+x1+box_w)/2)
            if frames == 1 or frames % 200 == 0 or obj_id not in way.keys():
                way[obj_id] = []

            way[obj_id].append([cx-5, y1+box_h-5, cx+5, y1+box_h-10])
            if len(way[obj_id]) > 0:
                if cx not in way[obj_id][-1]:
                    way[obj_id].append([cx, y1 + box_h - 5, cx, y1 + box_h - 10])
            else:
                way[obj_id].append([cx, y1 + box_h - 5, cx, y1 + box_h - 10])

            for temp_cx1, temp_y1, temp_cx2, temp_y2 in way[obj_id]:
                cv2.line(frame, (temp_cx1, temp_y1), (temp_cx2, temp_y2), color, 5)
            cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    # cv2.imshow('Stream', frame)
    outvideo.write(frame)
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break

totaltime = time.time()-starttime
print(frames, "frames", totaltime/frames, "s/frame")
cv2.destroyAllWindows()
outvideo.release()
