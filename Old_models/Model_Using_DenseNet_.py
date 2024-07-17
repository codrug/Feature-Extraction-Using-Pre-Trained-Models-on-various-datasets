# Import necessary libraries
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Get current directory
current_directory = os.getcwd()
print("Current Directory:", current_directory)

# Download dataset using Kaggle API (ensure Kaggle API is set up correctly)
!pip install kaggle
dataset_name = 'jaidalmotra/weed-detection'
os.system(f'kaggle datasets download -d {dataset_name}')

# Unzip the downloaded dataset
import zipfile

zip_file_path = 'weed-detection.zip'
extract_dir = current_directory

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Load training dataset
train_directory = os.path.join(current_directory, 'train')

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = datagen.flow_from_directory(
    train_directory,
    class_mode='categorical',
    target_size=(256, 256),
    batch_size=32
)

# Load test dataset
test_directory = os.path.join(current_directory, 'test')

test_data = datagen.flow_from_directory(
    test_directory,
    class_mode='categorical',
    target_size=(256, 256),
    batch_size=32
)

# Visualize the data
directory_path = os.path.join(current_directory, 'train')
allowed_extensions = ('.jpeg', '.jpg', '.png')

file_list = [
    file_name for file_name in os.listdir(directory_path)
    if os.path.splitext(file_name)[-1].lower() in allowed_extensions
]

for file_name in file_list[:5]:
    img_path = os.path.join(directory_path, file_name)
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.show()

# Define the model
conv_base = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(256, 256, 3),
    pooling='avg'
)

conv_base.trainable = False
conv_base.summary()

model = Sequential()
model.add(conv_base)
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.35))
model.add(BatchNormalization())
model.add(Dense(120, activation='relu'))
model.add(Dense(1, activation='softmax'))

model.summary()

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Train the model
history = model.fit(
    train_data,
    epochs=10,
    validation_data=test_data,
    callbacks=[early_stopping]
)

# Evaluate the model
evaluation = model.evaluate(test_data)
print("Validation Loss:", evaluation[0])
print("Validation Accuracy:", evaluation[1])

# Plot training metrics
def plot_metrics(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

plot_metrics(history)


