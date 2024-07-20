import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


# Paths to the dataset
img_dir = '/content/drive/MyDrive/crop-and-weed-detection-DatasetNinja/data/images/img'
ann_dir = '/content/drive/MyDrive/crop-and-weed-detection-DatasetNinja/data/annotations/ann'

# Load images and annotations
def load_data(img_dir, ann_dir):
    images = []
    labels = []
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        ann_path = os.path.join(ann_dir, img_name + '.json')

        # Load image
        img = Image.open(img_path)
        img = img.resize((224, 224))
        images.append(np.array(img) / 255.0)  # Normalize to [0, 1]

        # Load annotation and set label (1 for weed, 0 for crop)
        with open(ann_path, 'r') as f:
            ann = json.load(f)
        if any(obj['classTitle'] == 'weed' for obj in ann['objects']):
            labels.append(1)
        else:
            labels.append(0)

    return np.array(images), np.array(labels)


# Load the dataset
images, labels = load_data(img_dir, ann_dir)


X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)


# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Data augmentation for validation data (only rescaling)
val_datagen = ImageDataGenerator()

# Create data generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
val_generator = val_datagen.flow(X_val, y_val, batch_size=32)

# Load VGG16 model pre-trained on ImageNet, excluding top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Adding custom top layers for binary classification
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('weed_detection_VGG-16_.keras', save_best_only=True, monitor='val_loss')

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[early_stopping, model_checkpoint]
)

# Saving the model
model.save('weed_detection_VGG-16_.keras')



# Saving the model
model.save('weed_detection_VGG-16_.keras')

# Load the best model
model = tf.keras.models.load_model('weed_detection_VGG-16_.keras')

# Evaluate the model
train_loss, train_acc = model.evaluate(train_generator)
val_loss, val_acc = model.evaluate(val_generator)

print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")



# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
