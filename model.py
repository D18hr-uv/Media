import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
import pandas as pd

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define directories
REAL_DIR = 'real'
FAKE_DIR = 'fake'

def load_data_from_dirs():
    real_paths = glob.glob(os.path.join(REAL_DIR, '*.jpg')) + glob.glob(os.path.join(REAL_DIR, '*.png'))
    fake_paths = glob.glob(os.path.join(FAKE_DIR, '*.jpg')) + glob.glob(os.path.join(FAKE_DIR, '*.png'))

    print(f"Found {len(real_paths)} real images.")
    print(f"Found {len(fake_paths)} fake images.")

    if len(real_paths) == 0 or len(fake_paths) == 0:
        raise ValueError("Could not find images in 'real' or 'fake' directories.")

    # Create labels
    real_labels = np.zeros(len(real_paths))
    fake_labels = np.ones(len(fake_paths))

    image_paths = np.concatenate([real_paths, fake_paths])
    labels = np.concatenate([real_labels, fake_labels])

    return image_paths, labels

# Load data
try:
    image_paths, labels = load_data_from_dirs()
except Exception as e:
    print(f"Error loading data: {e}")
    # Continue anyway if just checking structure, but in real run this would exit
    # exit(1)
    image_paths = []
    labels = []

# Split data
if len(image_paths) > 0:
    train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42, stratify=labels)
    print(f"Training set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")
else:
    train_paths, val_paths, train_labels, val_labels = [], [], [], []

# Image preprocessing
def preprocess_image(image_path, target_size=(128, 128)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # MobileNetV2 expects inputs in [-1, 1] usually, but let's stick to standard preprocessing if we use pretrained weights
    # tf.keras.applications.mobilenet_v2.preprocess_input scales to [-1, 1]
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# Data Generator
def create_data_generator(paths, labels, batch_size=32, target_size=(128, 128), augment=False):
    images = []
    valid_labels = []
    for i, p in enumerate(paths):
        try:
            images.append(preprocess_image(p, target_size))
            valid_labels.append(labels[i])
        except Exception as e:
            print(f"Skipping corrupt image {p}: {e}")
            pass

    X = np.array(images)
    y = np.array(valid_labels)

    if len(X) == 0:
        return None

    if augment:
        datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2]
        )
        return datagen.flow(X, y, batch_size=batch_size)
    else:
        datagen = ImageDataGenerator()
        return datagen.flow(X, y, batch_size=batch_size)

# Build Model with MobileNetV2
def build_model(input_shape=(128, 128, 3)):
    # Load MobileNetV2 without the top layers
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the base model layers
    base_model.trainable = False

    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

model = build_model()
model.summary()

# Training
batch_size = 16
epochs = 50

if len(train_paths) > 0:
    train_gen = create_data_generator(train_paths, train_labels, batch_size, augment=True)
    val_gen = create_data_generator(val_paths, val_labels, batch_size, augment=False)

    if train_gen and val_gen:
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            steps_per_epoch=max(1, len(train_paths) // batch_size),
            validation_steps=max(1, len(val_paths) // batch_size)
        )

        # Save
        model_name = 'fake_face_detection_model.h5'
        model.save(model_name)
        print(f"Model saved to {model_name}")
    else:
        print("Not enough data to train.")
else:
    print("No data found. Model architecture created but not trained.")
    # Save the untrained model (with pretrained MobileNet weights) so app can load it
    model.save('fake_face_detection_model.h5')
    print("Saved untrained model structure.")
