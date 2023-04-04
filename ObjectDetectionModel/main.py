import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from ppadb.client import Client as AdbClient
from IPython.display import display
import pandas as pd
# client = AdbClient(host="127.0.0.1", port=5037)
# device = client.device("RF8NA2BRHKJ")

model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='yolov5/runs/train/exp7/weights/best.pt', force_reload=True)

# results = model("photo.jpg")

# device.shell(" am start -a android.media.action.IMAGE_CAPTURE && input keyevent 27")
# device.shell("mv /storage/self/primary/DCIM/Camera/*.jpg /storage/self/primary/DCIM/Camera/image.jpg")
# device.pull("/storage/self/primary/DCIM/Camera/image.jpg", "./image.jpg")
results = model("photo.jpg")
# device.shell("rm /storage/self/primary/DCIM/Camera/image.jpg")
# list_for_item1 =  results.xyxy
list_of_items_xyxy = results.xyxy[0].tolist()  # 15 glass , 16 can , 17 plastic
print(list_of_items_xyxy[0])

image = np.squeeze(results.render())
image = cv2.resize(image, (640, 480))
cv2.imshow('a', image)
cv2.waitKey(0)
