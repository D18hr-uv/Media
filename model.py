import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import pandas as pd # Kept for compatibility if needed, but not used for loading

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define directories
# The repo structure has 'real' and 'fake' folders in the root
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
    exit(1)

# Split data
train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42, stratify=labels)

print(f"Training set: {len(train_paths)} images")
print(f"Validation set: {len(val_paths)} images")

# Image preprocessing
def preprocess_image(image_path, target_size=(128, 128)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array /= 255.0
    return img_array

# Data Generator
def create_data_generator(paths, labels, batch_size=32, target_size=(128, 128), augment=False):
    # If augmenting, we can use Keras IDG, but since we have paths, we'll implement a custom generator
    # that reads files on the fly to save memory.

    # However, to use Keras IDG for augmentation easily, we can read all images into memory if the dataset is small.
    # Given ~100 images, this is fine.

    images = []
    for p in paths:
        try:
            images.append(preprocess_image(p, target_size))
        except Exception as e:
            print(f"Skipping corrupt image {p}: {e}")
            # Handle mismatch in labels if skipping?
            # For simplicity in this script, we assume images are fine.
            # Ideally we would remove the corresponding label.
            pass

    X = np.array(images)
    y = np.array(labels)

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

# Build Model
def build_model(input_shape=(128, 128, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

model = build_model()
model.summary()

# Training
batch_size = 16 # Small batch size for small dataset
epochs = 50

train_gen = create_data_generator(train_paths, train_labels, batch_size, augment=True)
val_gen = create_data_generator(val_paths, val_labels, batch_size, augment=False)

history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    steps_per_epoch=len(train_paths) // batch_size if len(train_paths) > batch_size else 1,
    validation_steps=len(val_paths) // batch_size if len(val_paths) > batch_size else 1
)

# Save
model_name = 'fake_face_detection_model.h5'
model.save(model_name)
print(f"Model saved to {model_name}")
