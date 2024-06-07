import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from PIL import Image, ImageChops, ImageEnhance
import os
import random
import pickle

# Create a directory for temporary files if it doesn't exist
temp_dir = 'temp'
os.makedirs(temp_dir, exist_ok=True)

def convert_to_ela_image(path, quality=90):
    temp_dir = 'temp'
    os.makedirs(temp_dir, exist_ok=True)  # Create the temp directory if it doesn't exist
    temp_filename = os.path.join(temp_dir, 'temp_file_name.jpg')
    ela_filename = os.path.join(temp_dir, 'temp_ela.png')

    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image

image_size = (128, 128)

def prepare_image(image_path):
    ela_image = convert_to_ela_image(image_path, 90)
    ela_image = ela_image.resize(image_size)
    return np.array(ela_image) / 255.0

X = []
Y = []

# Load real images
path_real = "CASIA2 Dataset/Au"
for dirname, _, filenames in os.walk(path_real):
    for filename in filenames:
        if filename.endswith(('jpg', 'png')):
            full_path = os.path.join(dirname, filename)
            X.append(prepare_image(full_path))
            Y.append(1)
            if len(Y) % 500 == 0:
                print(f'Processing {len(Y)} images')

random.shuffle(X)
X = X[:2100]
Y = Y[:2100]

# Load fake images
path_fake = 'CASIA2 Dataset/Tp'
for dirname, _, filenames in os.walk(path_fake):
    for filename in filenames:
        if filename.endswith(('jpg', 'png')):
            full_path = os.path.join(dirname, filename)
            X.append(prepare_image(full_path))
            Y.append(0)
            if len(Y) % 500 == 0:
                print(f'Processing {len(Y)} images')

X = np.array(X)
Y = to_categorical(Y, 2)
X = X.reshape(-1, 128, 128, 3)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=5)

def build_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='valid', activation='relu', input_shape=(128, 128, 3)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    return model

model = build_model()
model.summary()

epochs = 10
batch_size = 32

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, Y_val))

pickle.dump(model,open("model.pkl","wb"))
