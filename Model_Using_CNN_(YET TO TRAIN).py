# Import necessary libraries
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.callbacks as keras_callbacks

# Data Preparation
train_directory = r'C:\Users\dhruv\Model_dev_yt\train'
test_directory = r'C:\Users\dhruv\Model_dev_yt\test'

# Image Data Generators with Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# Create Data Generators
train_data = train_datagen.flow_from_directory(
    train_directory,
    class_mode='categorical',
    target_size=(256, 256),
    batch_size=32
)

test_data = test_datagen.flow_from_directory(
    test_directory,
    class_mode='categorical',
    target_size=(256, 256),
    batch_size=32
)

# Data Visualization
def plot_images_from_directory(directory, num_images=5):
    class_directories = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    images_plotted = 0

    for class_dir in class_directories:
        file_list = [
            os.path.join(class_dir, file_name) for file_name in os.listdir(class_dir)
            if os.path.splitext(file_name)[-1].lower() in ('.jpeg', '.jpg', '.png')
        ]

        for img_path in file_list[:num_images]:
            img = mpimg.imread(img_path)
            plt.imshow(img)
            plt.title(f"Class: {os.path.basename(class_dir)}")
            plt.axis('off')
            plt.show()
            images_plotted += 1

            if images_plotted >= num_images:
                break
        if images_plotted >= num_images:
            break

print("Sample Training Images:")
plot_images_from_directory(train_directory)

# Custom CNN Model Definition
model = Sequential([
    Input(shape=(256, 256, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),

    Dense(train_data.num_classes, activation='softmax')
])

# Model Compilation
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = keras_callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=10,  # Increased patience for a more thorough training
    verbose=1,
    mode="auto",
    restore_best_weights=True
)

model_checkpoint = keras_callbacks.ModelCheckpoint(
    "best_model.h5",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

callbacks = [early_stopping, model_checkpoint]

# Model Training
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=50,  # Increased number of epochs
    callbacks=callbacks
)

# Model Evaluation
evaluation = model.evaluate(test_data)
print("Validation Loss:", evaluation[0])
print("Validation Accuracy:", evaluation[1])

# Plotting Training Metrics
def plot_metrics(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

plot_metrics(history)



