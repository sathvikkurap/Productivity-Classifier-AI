import os
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from PIL import Image

# Hyperparameters
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 2  # Productive and non-productive
MODEL_SAVE_PATH = 'saved_model/model.h5'

# Path to dataset directories (productive and non-productive folders)
productive_path = "dataset/productive/"
non_productive_path = "dataset/non_productive/"

# Helper function to load images and resize them to 128x128 with consistent channels (RGB)
def load_images_from_directory(directory_path):
    images = []
    for filename in os.listdir(directory_path):
        img_path = os.path.join(directory_path, filename)
        img = Image.open(img_path).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        
        # Convert image to RGB if it is not already in that mode
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
        images.append(img)
    
    return np.array(images)

# Load the dataset
productive_images = load_images_from_directory(productive_path)
non_productive_images = load_images_from_directory(non_productive_path)

# Create labels for productive and non-productive tasks
productive_labels = np.array([[1, 0]] * productive_images.shape[0])  # [1, 0] for productive
non_productive_labels = np.array([[0, 1]] * non_productive_images.shape[0])  # [0, 1] for non-productive

# Combine the data and labels
all_data = np.concatenate([productive_images, non_productive_images], axis=0)
all_labels = np.concatenate([productive_labels, non_productive_labels], axis=0)

# Shuffle the data
indices = np.arange(all_data.shape[0])
np.random.shuffle(indices)
all_data = all_data[indices]
all_labels = all_labels[indices]

# Split the data into training and validation sets
train_size = int(0.8 * len(all_data))  # 80% for training
x_train, x_val = all_data[:train_size], all_data[train_size:]
y_train, y_val = all_labels[:train_size], all_labels[train_size:]

# Define the CNN model
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (5, 5), activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (5, 5), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Create the model
model = build_model()

# Augmenting data to help the model generalize better
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)

# Fit the model on the training data with data augmentation
history = model.fit(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                    epochs=EPOCHS,
                    validation_data=(x_val, y_val))

# Save the trained model
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# Evaluate the model on validation data
val_loss, val_acc = model.evaluate(x_val, y_val)
print(f"Validation accuracy: {val_acc:.4f}, Validation loss: {val_loss:.4f}")

