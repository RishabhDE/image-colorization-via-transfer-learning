import tensorflow as tf
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt

# Load and preprocess image
def load_and_preprocess_image(image_path, target_size, is_grayscale=False):
    img = tf.io.read_file(image_path)
    channels = 1 if is_grayscale else 3
    img = tf.image.decode_jpeg(img, channels=channels)
    img = tf.image.resize(img, target_size)
    img = img / 255.0  # Normalize to [0, 1]
    return img

# Convert RGB image to LAB using TensorFlow
def convert_to_lab(image):
    image_np = image.numpy() * 255.0  # Convert to 0-255 range for OpenCV
    image_np = image_np.astype(np.uint8)
    lab_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    lab_image = lab_image / 255.0  # Normalize to 0-1
    return lab_image

# Wrapper to use convert_to_lab in TensorFlow pipeline
@tf.function
def preprocess_lab_image(image):
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)  # Convert to uint8 for OpenCV
    lab_image = tf.py_function(func=convert_to_lab, inp=[image], Tout=tf.float32)
    lab_image.set_shape([None, None, 3])  # Set shape to avoid shape issues in TensorFlow
    return lab_image

# Load and preprocess images
def load_and_preprocess(gray_path, lab_path, target_size):
    gray_img = load_and_preprocess_image(gray_path, target_size, is_grayscale=True)
    rgb_img = load_and_preprocess_image(lab_path, target_size, is_grayscale=False)
    lab_img = preprocess_lab_image(rgb_img)
    return gray_img, lab_img

# Create dataset for test images
def create_test_dataset(gray_image_paths, target_size=(256, 256), batch_size=32):
    gray_image_paths = tf.constant(gray_image_paths, dtype=tf.string)
    
    # Define the dataset from image paths
    dataset = tf.data.Dataset.from_tensor_slices(gray_image_paths)
    
    # Map the dataset to preprocess images
    dataset = dataset.map(lambda gray_path: load_and_preprocess_image(gray_path, target_size, is_grayscale=True), num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Define directories
black_test_dir = 'data/test_black'  # Update with your actual path
color_test_dir = 'data/test_color'   # Update with your actual path

# Load image paths
black_images_test_paths = glob.glob(os.path.join(black_test_dir, '*.jpg'))[:5]
color_images_test_paths = glob.glob(os.path.join(color_test_dir, '*.jpg'))[:5]

# Ensure these paths are not empty
assert len(black_images_test_paths) > 0, "No black and white test images found."
assert len(color_images_test_paths) > 0, "No color test images found."

# Create the test dataset
test_dataset = create_test_dataset(black_images_test_paths, target_size=(256, 256))

# Load the trained generator model
generator = tf.keras.models.load_model('pix2pix_model_generator.keras')

# Function to generate colorful images
def generate_images(model, test_input):
    prediction = model(test_input, training=False)  # Set training=False for inference
    return prediction

# Function to show images
def show_images(gray_images, color_images, num_images=5):
    plt.figure(figsize=(18, 8))

    for i in range(num_images):
        # Display generated color images
        plt.subplot(2, num_images, i + 1)
        plt.imshow(tf.squeeze(gray_images[i]), cmap='gray')
        plt.axis('off')
        plt.title('Generated Colorful Image')

        # Display original color image
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(tf.squeeze(color_images[i]))
        plt.axis('off')
        plt.title('Original Color Image')

    plt.show()

# Iterate through the test dataset and generate colorful images
generated_images_list = []
original_images_list = []

for gray_img_batch in test_dataset:
    generated_images = generate_images(generator, gray_img_batch)
    
    # Since test_dataset might be batched, take images from the first batch
    for i in range(generated_images.shape[0]):
        generated_images_list.append(generated_images[i])

        # Match the color image paths to generated images
        original_color_img_path = color_images_test_paths[i % len(color_images_test_paths)]  # Use modulo to avoid out-of-bounds error
        original_color_img = load_and_preprocess_image(original_color_img_path, target_size=(256, 256), is_grayscale=False)
        original_images_list.append(original_color_img)

# Display images using the defined function
show_images(generated_images_list, original_images_list, num_images=len(generated_images_list))
