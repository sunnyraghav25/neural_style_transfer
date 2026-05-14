import sys
try:
    import tensorflow as tf
except ModuleNotFoundError:
    print("Error: TensorFlow is not installed. Install it with 'pip install tensorflow'.")
    sys.exit(1)
import tensorflow_hub as hub
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import os

# --- MODEL INITIALIZATION ---
# Using pre-trained Magenta model for Task-3
def load_model():
    print("Fetching model from TensorFlow Hub...")
    return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def preprocess_image(image_path):
    """Loads and scales image to float32 tensor."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img[tf.newaxis, :]

def run_task():
    # File validation
    if not os.path.exists('content.jpg') or not os.path.exists('style.jpg'):
        print("Check: 'content.jpg' and 'style.jpg' must be in the folder.")
        return

    try:
        model = load_model()
        
        # Load tensors
        content = preprocess_image('content.jpg')
        style = preprocess_image('style.jpg')

        print("Generating stylized output...")
        stylized_image = model(tf.constant(content), tf.constant(style))[0]

        # Display Result
        plt.figure(figsize=(10, 10))
        plt.imshow(np.squeeze(stylized_image))
        plt.title("Neural Style Transfer - Intern Task 3")
        plt.axis('off')
        plt.show()

        # Save for submission
        output = PIL.Image.fromarray(np.uint8(np.squeeze(stylized_image)*255))
        output.save('task3_final_output.jpg')
        print("Success: saved as 'task3_final_output.jpg'")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_task()