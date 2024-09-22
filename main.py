import tensorflow as tf
import numpy as np
from PIL import Image
import time
import mss
import cv2
import os

# Hyperparameters
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
CHANNELS = 3
MODEL_SAVE_PATH = 'saved_model/model.h5'
NUM_CLASSES = 2  # productive and non-productive
SCREEN_CAPTURE_INTERVAL = 5  # seconds

# Load the trained model
model = tf.keras.models.load_model(MODEL_SAVE_PATH)

# Function to preprocess the screen capture to fit the model input
def preprocess_screen_capture(image):
    image = Image.fromarray(image)
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    image = np.array(image).astype(np.float32) / 255.0  # Normalize to [0,1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to classify the screen content
def classify_screen(image):
    processed_image = preprocess_screen_capture(image)
    
    # Run the model on the processed image
    predictions = model.predict(processed_image)
    prediction = predictions[0]
    
    if prediction[0] > prediction[1]:
        print("Detected: Productive task.")
    else:
        print("Detected: Non-productive task!")
        os.system('play -nq -t alsa synth 0.1 sine 440')  # Beep on Linux or replace for other OS

# Function to capture the screen periodically
def start_screen_monitoring():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Full screen

        while True:
            screenshot = sct.grab(monitor)
            image = np.array(screenshot)  # Convert MSS screenshot to NumPy array

            # Convert the image from BGRA to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

            # Classify the current screen content
            classify_screen(image)

            # Wait for the next screen capture
            time.sleep(SCREEN_CAPTURE_INTERVAL)

# Start the real-time screen monitoring
if __name__ == '__main__':
    start_screen_monitoring()
