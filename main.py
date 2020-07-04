import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# load model
model = load_model("model/RockPaperScissors.keras")

cp = cv2.VideoCapture(0)

while True:
    _, img = cp.read()

    # decode image
    cv2.imshow("rock", img)
    img = cv2.resize(img, (50, 50))
    img = np.array(img)
    img = image.array_to_img(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    # list labels
    labels = ['paper', 'rock', 'scissors']

    classes = model.predict(images)
    classes = np.argmax(classes)
    print(labels[classes])

    if cv2.waitKey(1) == ord("q"):
        break
