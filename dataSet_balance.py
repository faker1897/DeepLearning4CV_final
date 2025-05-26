import os
import tensorflow as tf
directory = "dataSet/train"

# Calculate the class_weight
class_counts = {}
for class_name in os.listdir(directory):
    class_path = os.path.join(directory, class_name)
    if os.path.isdir(class_path):
        num_images = len(os.listdir(class_path))
        class_counts[class_name] = num_images

print(class_counts)

print(tf.config.list_physical_devices('GPU'))
print(tf.__version__)
