
from Video import VideoShow
from Video import VideoGet
from Video import CountsPerSec
import argparse
from datetime import datetime
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
# from ppadb.client import Client as AdbClient
# from IPython.display import display
import pandas as pd
from threading import Thread

# client = AdbClient(host="127.0.0.1", port=5037)
# device = client.device("RF8NA2BRHKJ")

BottleModel = torch.hub.load('ultralytics/yolov5', 'custom',
                             path='ObjectDetectionModel/yolov5/runs/train/exp7/weights/best.pt', force_reload=True)


BoxModel = torch.hub.load('ultralytics/yolov5', 'custom',
                          path='BoxDetectionModel/yolov5/runs/train/exp/weights/best.pt', force_reload=True)
# making list of items pixels and classes
# list_of_items_xyxy = results.xyxy[0].tolist()  # 15 glass , 16 can , 17 plastic
# print(list_of_items_xyxy[0])

# bottleimage = np.squeeze(results.render())
# bottleimage = cv2.resize(bottleimage, (640, 480))
# cv2.imshow('a', bottleimage)
# cv2.waitKey(0)

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    image = cv2.resize(frame, (640, 480))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess the bottleimage to match the BottleModel's input format
    bottleimage = BottleModel(image)
    boximage = BoxModel(image)
    boximage = np.squeeze(boximage.render())

    bottleimage = np.squeeze(bottleimage.render())
    bottleimage = cv2.cvtColor(bottleimage, cv2.COLOR_BGR2RGB)
    boximage = cv2.cvtColor(boximage, cv2.COLOR_BGR2RGB)
    # Make a prediction using the BottleModel
    # Postprocess the prediction and display the result
    bottleimage = cv2.resize(bottleimage, (1080, 720))
    boximage = cv2.resize(boximage, (1080, 720))

    cv2.imshow('Bottle', bottleimage)
    cv2.imshow('Box', boximage)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
