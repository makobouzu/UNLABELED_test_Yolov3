import cv2
import numpy as np
import torch
import torchvision

class utils:
    @staticmethod
    def load_model():
        device = "cuda" if torch.cuda.is_available() else "cpu"
#        model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
        model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
        model.to(device).eval()
        return model
    
    @staticmethod
    def grab_frame(cap):
        _, frame = cap.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def get_pred(img, model):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        preprocess = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        input_tensor = preprocess(img).unsqueeze(0)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)["out"]
            output = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
        
        return output
