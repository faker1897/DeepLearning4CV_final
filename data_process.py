import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications.densenet import layers
from tensorflow import keras
from keras import Sequential


# show augmentation output
def show_augmentation(train_ds,data_augmentation):
    for images, labels in train_ds.take(1):
        original_images = images
        augmented_images = data_augmentation(images)
        break

    plt.figure(figsize=(10, 4))

    for i in range(5):
        # original_version
        plt.subplot(2, 5, i + 1)
        plt.imshow(tf.squeeze(original_images[i]/255), cmap='gray')
        plt.title("Original")
        plt.axis("off")
        # augmented_version
        plt.subplot(2, 5, i + 6)
        plt.imshow(tf.squeeze(augmented_images[i]), cmap='gray')
        plt.title("Augmented")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def to_Hot(image,label):
    label = tf.one_hot(label, depth=7)
    return image, label

rescale = keras.Sequential([
    layers.Rescaling(1. / 255),
])

data_augmentation = keras.Sequential([

    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.1),
    # Normalization combined with random brightness adjustment improves the model's robustness
    layers.RandomBrightness(factor=0.1),
])

'''
index: indicates which class to apply augmentation to;
rand: controls the proportion of samples to be augmented.
'''
def augment_combined(images, labels):
    images = rescale(images)
    indices = tf.argmax(labels, axis=1)
    rand_vector = tf.random.uniform(shape=[tf.shape(images)[0]])
    # disgust 60% chance of being augmented
    mask1 = tf.logical_and(tf.equal(indices, 1), rand_vector < 0.6)
    # fear 30% chance of being augmented
    mask2 = tf.logical_and(tf.equal(indices, 2), rand_vector < 0.3)
    # surprise 10% chance of being augmented
    mask3 = tf.logical_and(tf.equal(indices, 6), rand_vector < 0.1)
    # angry 20% chance of being augmented
    mask4 = tf.logical_and(tf.equal(indices, 0), rand_vector < 0.2)
    # Combine all masks into a single one
    mask = tf.logical_or(tf.logical_or(mask1, mask2), tf.logical_or(mask3,mask4))
    mask = tf.reshape(mask, (-1, 1, 1, 1))
    augmented = data_augmentation(images)
    # Use the mask to select augmented images where applicable
    final_images = tf.where(mask, augmented, images)

    return final_images, labels
