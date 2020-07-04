from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard

"""
command for split data set to test, val, training
sf.ratio("dataset/", "data/", seed=1337, ratio=(.8, .1, .1))
"""

# get data frame for train, val, test
trainingPath = "data/train"
valPath = "data/val"
testPath = "data/test"

# make image decoder
imageDecode = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    horizontal_flip=True,
    shear_range=0.2
)

# decode all image for training, val, and test
trainImage = imageDecode.flow_from_directory(
    trainingPath,
    target_size=(50, 50),
    batch_size=4,
    class_mode="categorical"
)

valImage = imageDecode.flow_from_directory(
    valPath,
    target_size=(50, 50),
    batch_size=4,
    class_mode="categorical"
)

testImage = imageDecode.flow_from_directory(
    testPath,
    target_size=(50, 50),
    batch_size=4,
    class_mode="categorical"
)

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

# compile model
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(
    loss="mean_squared_logarithmic_error",
    optimizer=sgd,
    metrics=['accuracy']
)

# training model
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

model.save("model/RockPaperScissors.keras", include_optimizer=False)
