# Image-Classification-RockScissorsPaper-

**Link Dataset For Training Model**
https://dicodingacademy.blob.core.windows.net/picodiploma/ml_pemula_academy/rockpaperscissors.zip

**Explanation of each file**

**1. training.py**

For training model ai

**Step by Step training model**

Make image decoder
```bash
imageDecode = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    horizontal_flip=True,
    shear_range=0.2
)
```

Decode image for training model
```bash
trainImage = imageDecode.flow_from_directory(
    trainingPath,
    target_size=(50, 50),
    batch_size=4,
    class_mode="categorical"
)
```

Make model for ML
```bash
# create model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(50, 50, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(512, activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(3, activation="sigmoid"))
```

Compile Model
```bash
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(
    loss="mean_squared_logarithmic_error",
    optimizer=sgd,
    metrics=['accuracy']
)
```

Training Model
```bash
newCallback = TensorBoard(log_dir="./graph", histogram_freq=0, write_graph=True, write_images=True)

model.fit(
    trainImage,
    steps_per_epoch=50,
    epochs=30,
    validation_data=valImage,
    validation_steps=25,
    verbose=2,
    callbacks=[newCallback]
)
```

Save model
```bash
model.save("model/RockPaperScissors.keras", include_optimizer=False)
```

**2. Main.py**

Application will access webcam from pc or laptop, image from webcam will be decode to array and proccess with model for check classes from image

Load Model ML
```bash
model = load_model("model/RockPaperScissors.keras")
```

Access webcam
```bash
cp = cv2.VideoCapture(0)
```

Read Image from webcam
```bash
_, img = cp.read()
```

Decode image
```bash
cv2.imshow("rock", img)
img = cv2.resize(img, (50, 50))
img = np.array(img)
img = image.array_to_img(img)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
```

Predict image classes with model
```bash
labels = ['paper', 'rock', 'scissors']

classes = model.predict(images)
classes = np.argmax(classes)
print(labels[classes])
```
