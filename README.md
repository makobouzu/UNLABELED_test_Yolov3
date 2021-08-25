# UNLABELED_test_Yolov3

Test PC
- Mac Big Sur 11.5.1


```
python run.py
```

If you want to test with video
```run.py
cap = cv2.VideoCapture("mov/test.mov")

_, image = cap.read()
image = cv2.resize(image, (640, 480)) # ADD
bbox = get_bbox(yolo_model, image, device, args.obj_thold, args.nms_thold, args.model_res)
```
