import torch
from yolov3.utils import detect, get_bbox
from yolov3.darknet import Darknet
from yolov3.head import Yolo3
import argparse
import cv2
import numpy as np
from deeplab.utils import *
from pythonosc import osc_message_builder
from pythonosc import udp_client

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cline = argparse.ArgumentParser(description='YOLO v3 webcam detection demo')
cline.add_argument('-weights', default='yolov3/data/yolo3_weights.pth',
                   help='path to pretrained weights')
cline.add_argument('-obj_thold', type=float, default=0.65,
                   help='threshold for objectness value')
cline.add_argument('-nms_thold', type=float, default=0.4,
                   help='threshold for non max supression')
cline.add_argument('-model_res', type=int, default=416,
                   help='resolution of the model\'s input')

client = udp_client.SimpleUDPClient("127.0.0.1", 3296)

def make_osc(input_list):
    msg = osc_message_builder.OscMessageBuilder(address= "/bbox")
    for values in input_list:
        for value in values:
            msg.add_arg(value)
    msg = msg.build()
    return msg

if __name__ == "__main__":
    args = cline.parse_args()
    with torch.no_grad():
        bbone = Darknet()
        bbone = bbone.extractor
        yolo_model = Yolo3(bbone)

        print(f'Loading weights from {args.weights}')
        yolo_model.load_state_dict(torch.load(args.weights))
        yolo_model.to(device)

        deeplab_model = utils.load_model()

        cap = cv2.VideoCapture("mov/test.mov")

        while(True):
            _, image = cap.read()
            image = cv2.resize(image, (640, 480))
            bbox = get_bbox(yolo_model, image, device, args.obj_thold, args.nms_thold, args.model_res)
            osc_msg = make_osc(bbox)
            client.send_message("/index", len(bbox))
            client.send(osc_msg)
            
            mask = np.zeros_like(image)
            for box in bbox:
                cv2.rectangle(mask, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), thickness=-1)
            bbox_mask = cv2.bitwise_and(image, mask)

            labels = utils.get_pred(bbox_mask, deeplab_model)
            segment_mask = labels == 15
            segment_mask = np.repeat(segment_mask[:, :, np.newaxis], 3, axis = 2)

            output = (segment_mask * 255).astype("uint8")

            #original, bbox_mask, segmentation
            view = np.hstack((image, bbox_mask, output))
            cv2.imshow('webcam', view)

            k = cv2.waitKey(100)
            if k == 27:# Press Esc to quit
                break

        cap.release()
        cv2.destroyAllWindows()
