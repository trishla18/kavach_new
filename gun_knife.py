from roboflow import Roboflow
rf = Roboflow(api_key="vNuH3IiQvIqq0fvGssUE")
project = rf.workspace().project("gun-knife-thesis")
model = project.version(11).model

# infer on a local image
print(model.predict("your_image.jpg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())