import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import glob
from gan import Generator, Discriminator
from dataset import make_dataset


def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        return img

    preprocessed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # Concatenate images into a row
        if single_row.size == 0:
            single_row = preprocessed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocessed[b, :, :, :]), axis=1)

        # Concatenate rows into final image
        if (b + 1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    Image.fromarray(final_image).save(image_path)


def celoss_ones(logits):
    # Cross entropy loss for labels = 1 (using label smoothing 0.9)
    y = tf.ones_like(logits) * 0.9
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    # Cross entropy loss for labels = 0
    y = tf.zeros_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)


def d_loss_fn(generator, discriminator, batch_z, batch_x, training):
    # Discriminator loss
    fake_image = generator(batch_z, training)
    d_fake_logits = discriminator(fake_image, training)
    d_real_logits = discriminator(batch_x, training)
    d_loss_real = celoss_ones(d_real_logits)
    d_loss_fake = celoss_zeros(d_fake_logits)
    loss = d_loss_fake + d_loss_real
    return loss


def g_loss_fn(generator, discriminator, batch_z, training):
    # Generator loss
    fake_image = generator(batch_z, training)
    d_fake_logits = discriminator(fake_image, training)
    loss = celoss_ones(d_fake_logits)
    return loss


def main():
    tf.random.set_seed(100)
    np.random.seed(100)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    assert tf.__version__.startswith('2.')

    z_dim = 100  # latent vector length
    epochs = 1900  # number of training epochs
    batch_size = 64
    learning_rate = 0.0002
    training = True

    # Load image paths
    img_path = glob.glob(r'E:\America_homework\DeepLearning\DeepLearning4CV_final\dataSet\train\disgust\*.jpg') + \
               glob.glob(r'E:\America_homework\DeepLearning\DeepLearning4CV_final\dataSet\test\disgust\*.jpg')
    print('images num:', len(img_path))

    # Create dataset
    dataset, img_shape, _ = make_dataset(img_path, batch_size, resize=64)
    print(dataset, img_shape)
    sample = next(iter(dataset))
    print(sample.shape, tf.reduce_max(sample).numpy(), tf.reduce_min(sample).numpy())
    dataset = dataset.repeat()
    db_iter = iter(dataset)

    generator = Generator()
    generator.build(input_shape=(4, z_dim))
    discriminator = Discriminator()
    discriminator.build(input_shape=(4, 64, 64, 1))

    g_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    d_losses, g_losses = [], []
    for epoch in range(epochs):
        # Train discriminator
        for _ in range(5):
            batch_z = tf.random.normal([batch_size, z_dim])
            batch_x = next(db_iter)
            with tf.GradientTape() as tape:
                d_loss = d_loss_fn(generator, discriminator, batch_z, batch_x, training)
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

            batch_z = tf.random.normal([batch_size, z_dim])
            batch_x = next(db_iter)
            with tf.GradientTape() as tape:
                g_loss = g_loss_fn(generator, discriminator, batch_z, training)
            grads = tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % 100 == 0:
            print(epoch, 'd-loss:', float(d_loss), 'g-loss:', float(g_loss))
            z = tf.random.normal([100, z_dim])
            fake_image = generator(z, training=False)
            img_path = os.path.join('gan_images', 'gan_images-%d.png' % epoch)
            save_result(fake_image.numpy(), 10, img_path, color_mode='L')

            d_losses.append(float(d_loss))
            g_losses.append(float(g_loss))

            generator.save_weights(f'generator.ckpt-{epoch}')
            discriminator.save_weights(f'discriminator-{epoch}.ckpt')


if __name__ == '__main__':
    main()