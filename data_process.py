import matplotlib.pyplot as plt
import tensorflow as tf

def augmentation(train_ds,data_augmentation):
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