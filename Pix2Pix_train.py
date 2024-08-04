import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
import glob
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from Data_processing_functions import *


# Define Data Directory
dir_path = 'data'
color_dir = os.path.join(dir_path, 'train_color')
black_dir = os.path.join(dir_path, 'train_black')

# List all images
color_images_paths = glob.glob(os.path.join(color_dir, '*.jpg'))
black_images_paths = glob.glob(os.path.join(black_dir, '*.jpg'))

# Create the original dataset
dataset = create_dataset(black_images_paths, color_images_paths, target_size=(256, 256))

# Create the dataset with combined grayscale and edge-detected images
dataset_with_edges = create_dataset_with_edges(dataset)


# Define hyperparameters
hyperparams = {
    'learning_rate': 2e-4,
    'beta_1': 0.5,
    'batch_size': 1,
    'epochs': 50,
    'dropout_rate': 0.5,
    'lambda_l1': 100,
    'batch_norm': True,
    'initial_filters': 64,
    'kernel_size': 4,
    'dropout': True,
    'num_layers': 8,
    'use_data_augmentation': True
}


# Define the optimizers
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparams['learning_rate'], beta_1=hyperparams['beta_1'])
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparams['learning_rate'], beta_1=hyperparams['beta_1'])

# Define the checkpoint directory
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

# Create model instances
generator = Generator()
discriminator = Discriminator()

# Create a checkpoint object
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Train the model
gen_losses, disc_losses, val_gen_losses = model_fit(dataset_with_edges, val_dataset_with_edges, 
                                                    hyperparams, checkpoint, checkpoint_prefix)

generator.save('pix2pix_model_generator.h5')
discriminator.save('pix2pix_model_discriminator.h5')


# Visualize losses
visualize_losses(gen_losses, disc_losses, val_gen_losses)



# Load the generator model
generator = tf.keras.models.load_model('pix2pix_model_generator.h5')

# Load the discriminator model
discriminator = tf.keras.models.load_model('pix2pix_model_discriminator.h5')

# # Load and preprocess new sample data
def load_and_preprocess_image(image_path, target_size=(256, 256), is_grayscale=True):
    image = Image.open(image_path).convert('L' if is_grayscale else 'RGB')
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    if is_grayscale:
        image = np.expand_dims(image, axis=-1)  # Add channel dimension for grayscale
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Load the generator model
generator = tf.keras.models.load_model('pix2pix_model_generator.h5')

# Load new sample image
new_image_path = black_images_paths[:10]
new_image = load_and_preprocess_image(new_image_path, target_size=(256, 256), is_grayscale=True)

# Predict using the generator
def predict_with_generator(model, input_image):
    prediction = model(input_image, training=False)
    return prediction

# Get prediction
predicted_image = predict_with_generator(generator, new_image)

# Display the result
def display_results(input_image, generated_image, is_grayscale=True):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Input Image')
    if is_grayscale:
        plt.imshow(input_image[0, :, :, 0], cmap='gray')
    else:
        plt.imshow(input_image[0])
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Generated Image')
    if is_grayscale:
        plt.imshow((generated_image[0, :, :, 0] + 1) / 2, cmap='gray')  # Adjust based on output range
    else:
        plt.imshow((generated_image[0] + 1) / 2)  # Adjust based on output range
    plt.axis('off')

    plt.show()

# Display results
display_results(new_image, predicted_image, is_grayscale=True)