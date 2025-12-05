import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define directories
REAL_DIR = 'real'
FAKE_DIR = 'fake'

def load_data_from_dirs():
    # Load .jpg and .png files
    real_paths = glob.glob(os.path.join(REAL_DIR, '*.jpg')) + glob.glob(os.path.join(REAL_DIR, '*.png'))
    fake_paths = glob.glob(os.path.join(FAKE_DIR, '*.jpg')) + glob.glob(os.path.join(FAKE_DIR, '*.png'))

    print(f"Found {len(real_paths)} real images.")
    print(f"Found {len(fake_paths)} fake images.")

    if len(real_paths) == 0 or len(fake_paths) == 0:
        print("Warning: One or both class directories are empty.")
        return [], []

    # Create labels: 0 for real, 1 for fake
    real_labels = np.zeros(len(real_paths))
    fake_labels = np.ones(len(fake_paths))

    image_paths = np.concatenate([real_paths, fake_paths])
    labels = np.concatenate([real_labels, fake_labels])

    return image_paths, labels

# Load data
image_paths, labels = load_data_from_dirs()

# Split data (stratified to ensure balance)
if len(image_paths) > 0:
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Training set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")
else:
    train_paths, val_paths = [], []
    train_labels, val_labels = [], []

# Image preprocessing function
def preprocess_image(image_path, target_size=(224, 224)): # EfficientNet likes 224x224
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # EfficientNetB0 expects 0-255 inputs and handles scaling internally via normalization layers in the model
    # typically, but let's stick to standard behavior.
    # However, keras.applications.efficientnet.preprocess_input does nothing for EfficientNetB0-B7!
    # It expects pixels in [0, 255].
    return img_array

# Data Generator
class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, paths, labels, batch_size=16, target_size=(224, 224), augment=False, shuffle=True):
        # Convert to numpy arrays to support indexing
        self.paths = np.array(paths)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(self.paths))

        if self.augment:
            self.datagen = ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=False,
                brightness_range=[0.8, 1.2],
                fill_mode='nearest'
            )

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.paths) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        # self.paths is a list, so we must use list comprehension or convert to array earlier
        # Here we convert to array in __init__ but to be safe let's assume it might be list
        batch_paths = [self.paths[i] for i in indices]
        batch_labels = self.labels[indices]

        X = np.zeros((len(batch_paths), *self.target_size, 3), dtype=np.float32)
        y = np.array(batch_labels, dtype=np.float32)

        for i, path in enumerate(batch_paths):
            try:
                img = preprocess_image(path, self.target_size)
                X[i] = img
            except Exception as e:
                print(f"Error loading {path}: {e}")

        if self.augment:
            # Augment images one by one or in batch
            # flow() takes 4D array
            X_gen = next(self.datagen.flow(X, batch_size=len(X), shuffle=False))
            return X_gen, y

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def build_model(input_shape=(224, 224, 3)):
    # Load EfficientNetB0
    # include_top=False means we load the feature extractor
    # weights='imagenet' loads pre-trained weights
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)

    # ---------------- STAGE 1: Train Head ----------------
    # Freeze the base model layers
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Use binary crossentropy
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model, base_model

model, base_model = build_model()
model.summary()

# Training Configuration
batch_size = 8 # Small batch size for better generalization on small data
epochs_stage_1 = 20
epochs_stage_2 = 20

if len(train_paths) > 0:
    train_gen = CustomDataGenerator(train_paths, train_labels, batch_size, augment=True)
    val_gen = CustomDataGenerator(val_paths, val_labels, batch_size, augment=False)

    # Callbacks
    checkpoint = ModelCheckpoint('fake_face_detection_model.h5',
                                monitor='val_accuracy',
                                save_best_only=True,
                                mode='max',
                                verbose=1)

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    print("\n=== STAGE 1: Training Head (Frozen Base) ===")
    history1 = model.fit(
        train_gen,
        epochs=epochs_stage_1,
        validation_data=val_gen,
        callbacks=[reduce_lr] # Don't save checkpoint yet, or maybe save simplistic one
    )

    # ---------------- STAGE 2: Fine Tuning ----------------
    print("\n=== STAGE 2: Fine Tuning (Unfrozen Top Layers) ===")

    # Unfreeze the last 30 layers
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Recompile with very low learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

    history2 = model.fit(
        train_gen,
        epochs=epochs_stage_2,
        validation_data=val_gen,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    print("Training Complete. Best model saved to fake_face_detection_model.h5")

else:
    print("No data found. Saving untrained model structure.")
    model.save('fake_face_detection_model.h5')
