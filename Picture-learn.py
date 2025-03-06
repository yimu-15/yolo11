from ultralytics import YOLO

# Load a model
model = YOLO("Models\\yolo11x.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model([r'D:\yolo11\JPEGlmages\2.png'], stream=True)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="Annotations\\2.jpg")  # save to diskpip install ultralytics