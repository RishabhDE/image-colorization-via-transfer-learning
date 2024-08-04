import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, UpSampling2D, concatenate, BatchNormalization, LeakyReLU, ReLU
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import time



# Define the downsampling and upsampling blocks
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(BatchNormalization())

    result.add(LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

    result.add(BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(hyperparams['dropout_rate']))

    result.add(ReLU())

    return result

# Define the Generator model
def Generator():
    inputs = Input(shape=[256, 256, 1])

    down_stack = []
    for i in range(hyperparams['num_layers']):
        filters = hyperparams['initial_filters'] * (2 ** i)
        down_stack.append(downsample(filters, hyperparams['kernel_size'], apply_batchnorm=hyperparams['batch_norm']))

    up_stack = []
    for i in range(hyperparams['num_layers']-1, 0, -1):
        filters = hyperparams['initial_filters'] * (2 ** i)
        up_stack.append(upsample(filters, hyperparams['kernel_size'], apply_dropout=hyperparams['dropout']))

    initializer = tf.random_normal_initializer(0., 0.02)
    last = Conv2DTranspose(2, hyperparams['kernel_size'], strides=2, padding='same', kernel_initializer=initializer, activation='tanh')

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concatenate([x, skip])

    x = last(x)

    return Model(inputs=inputs, outputs=x)

# Define the Discriminator model
def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = Input(shape=[256, 256, 1], name='input_image')
    tar = Input(shape=[256, 256, 2], name='target_image')

    x = concatenate([inp, tar])

    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)

    batchnorm1 = BatchNormalization()(conv)
    leaky_relu = LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    last = Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)

    return Model(inputs=[inp, tar], outputs=last)

# Loss functions
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (hyperparams['lambda_l1'] * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

# Training step function
def train_step(generator, discriminator, input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    return gen_total_loss, disc_loss

# Define the evaluation function
def evaluate_model(dataset, generator):
    total_gen_loss = 0
    num_batches = 0
    for input_image, target in dataset:
        gen_output = generator(input_image, training=False)
        gen_total_loss, _, _ = generator_loss(None, gen_output, target)
        total_gen_loss += gen_total_loss
        num_batches += 1
    avg_gen_loss = total_gen_loss / num_batches
    return avg_gen_loss

# Training function with validation loss
def model_fit(train_ds, val_ds, hyperparams, checkpoint, checkpoint_prefix):
    gen_losses = []
    disc_losses = []
    val_gen_losses = []

    generator = Generator()
    discriminator = Discriminator()

    # Define the optimizers
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparams['learning_rate'], beta_1=hyperparams['beta_1'])
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparams['learning_rate'], beta_1=hyperparams['beta_1'])

    for epoch in range(hyperparams['epochs']):
        start = time.time()
        epoch_gen_loss = 0
        epoch_disc_loss = 0

        # Progress bar
        progbar = tf.keras.utils.Progbar(len(train_ds), stateful_metrics=['loss'])
        
        for input_image, target in train_ds:
            gen_total_loss, disc_loss = train_step(generator, discriminator, input_image, target)
            epoch_gen_loss += gen_total_loss
            epoch_disc_loss += disc_loss

            # Update progress bar
            progbar.update(progbar.seen + 1, [('loss', gen_total_loss.numpy())])

        avg_gen_loss = epoch_gen_loss / len(train_ds)
        avg_disc_loss = epoch_disc_loss / len(train_ds)

        # Compute validation loss
        avg_val_gen_loss = evaluate_model(val_ds, generator)

        gen_losses.append(avg_gen_loss)
        disc_losses.append(avg_disc_loss)
        val_gen_losses.append(avg_val_gen_loss)

        # Save checkpoints every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f'Epoch: {epoch + 1}, Gen Loss: {avg_gen_loss.numpy()}, Disc Loss: {avg_disc_loss.numpy()}, Val Gen Loss: {avg_val_gen_loss.numpy()}')
        print(f'Time taken for epoch {epoch + 1}: {time.time() - start} sec\n')

    return gen_losses, disc_losses, val_gen_losses

# Visualize losses
def visualize_losses(gen_losses, disc_losses, val_gen_losses):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(gen_losses, label='Generator Loss (Train)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Loss over Time (Train)')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(disc_losses, label='Discriminator Loss (Train)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss over Time (Train)')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(val_gen_losses, label='Generator Loss (Val)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Loss over Time (Val)')
    plt.legend()
    
    plt.show()
