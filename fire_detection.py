from roboflow import Roboflow
import cv2
rf = Roboflow(api_key="vNuH3IiQvIqq0fvGssUE")
project = rf.workspace().project("fire-smoke-detection-odvk6")
model = project.version(1).model

# infer on a local image
print(model.predict("C:/Users/trish/Downloads/Fire-Smoke-Detection.v1i.yolov8/train/images/small_-66-_jpg.rf.a7372570635850bbbed201c96345ed63.jpg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("C:/Users/trish/Downloads/Fire-Smoke-Detection.v1i.yolov8/train/images/small_-66-_jpg.rf.a7372570635850bbbed201c96345ed63.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
#print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())