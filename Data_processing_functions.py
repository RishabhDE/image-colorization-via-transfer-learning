import tensorflow as tf
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
def load_image(image_path, target_size):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, target_size)
    img = img / 255.0
    return img

# Load RGB image and convert to LAB
def load_rgb_image(image_path, target_size):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = img / 255.0
    return img

# Convert RGB image to LAB using OpenCV
def convert_to_lab(image):
    image_np = image.numpy() * 255.0
    image_np = image_np.astype(np.uint8)
    lab_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    lab_image = lab_image.astype(np.float32) / 255.0
    return lab_image

# Wrapper to use convert_to_lab in TensorFlow pipeline
def preprocess_lab_image(image):
    lab_image = tf.py_function(func=convert_to_lab, inp=[image], Tout=tf.float32)
    lab_image.set_shape([None, None, 3])  # Set shape to avoid shape issues in TensorFlow
    return lab_image

# Combine grayscale image with edges
def combine_gray_and_edges(gray_img, lab_img):
    combined_images = []
    for i in range(gray_img.shape[0]):
        single_gray = gray_img[i, :, :, 0]
        edges = extract_edges(single_gray)
        combined = tf.stack([single_gray, edges], axis=-1)
        combined_images.append(combined)
    combined_batch = tf.stack(combined_images, axis=0)
    return combined_batch, lab_img

# Combine grayscale image with segments
def combine_gray_and_segment(gray_img, lab_img):
    combined_images = []
    for i in range(gray_img.shape[0]):
        single_gray = gray_img[i, :, :, 0]
        segment = otsu_segmentation(single_gray)
        combined = tf.stack([single_gray, segment], axis=-1)
        combined_images.append(combined)
    combined_batch = tf.stack(combined_images, axis=0)
    return combined_batch, lab_img

# Load and preprocess images
def load_and_preprocess(gray_path, lab_path, target_size):
    gray_img = load_image(gray_path, target_size)
    rgb_img = load_rgb_image(lab_path, target_size)
    lab_img = preprocess_lab_image(rgb_img)
    return gray_img, lab_img

# Create dataset
def create_dataset(gray_image_paths, lab_image_paths, target_size=(256, 256), batch_size=32):
    gray_image_paths = tf.constant(gray_image_paths, dtype=tf.string)
    lab_image_paths = tf.constant(lab_image_paths, dtype=tf.string)
    dataset = tf.data.Dataset.from_tensor_slices((gray_image_paths, lab_image_paths))
    dataset = dataset.map(lambda gray_path, lab_path: tf.py_function(
        func=lambda x, y: load_and_preprocess(x, y, target_size),
        inp=[gray_path, lab_path],
        Tout=[tf.float32, tf.float32]
    ), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Create dataset with edges
def create_dataset_with_edges(original_dataset):
    dataset_with_edges = original_dataset.map(
        lambda gray_img, lab_img: tf.py_function(
            func=combine_gray_and_edges,
            inp=[gray_img, lab_img],
            Tout=[tf.float32, tf.float32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset_with_edges

# Create dataset with segments
def create_dataset_with_segment(original_dataset):
    dataset_with_segment = original_dataset.map(
        lambda gray_img, lab_img: tf.py_function(
            func=combine_gray_and_segment,
            inp=[gray_img, lab_img],
            Tout=[tf.float32, tf.float32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset_with_segment