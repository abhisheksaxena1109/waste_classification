import ultralytics
import numpy
ultralytics.checks()
from ultralytics import YOLO
model = YOLO("model_weights.pt")
detection_output=model.predict(source=r"C:\Users\ABHISHEK SAXENA\Desktop\yolo_test\download.jpeg",conf=0.25,save=True)
print(detection_output)
print(detection_output[0].numpy())
