import tensorflow as tf
from keras.regularizers import l2
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense, Activation, \
    Flatten
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers,models
import os
import data_process
from sklearn.utils import class_weight
import numpy as np
import focal_loss

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Calculate the weight of each class
# Assume 0–6 correspond to angry, disgust, fear, happy, neutral, sad, and surprise.
class_labels = [0, 1, 2, 3, 4, 5, 6]
sample_counts = [3995, 436, 4097, 7215, 4965, 4830, 3171]
sum_samples = sum(sample_counts)
class_weight_dict = {}
for i in range(0,7):
    class_weight_dict[i] = sum_samples/(7*sample_counts[i])

print(class_weight_dict)

#Since the training and test sets have already been split, there's no need to set a seed.
# Load training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataSet/train",
    validation_split=0.2,
    subset="training",
    seed=529,
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=128,
)

# Load validation dataset
validate_ds = tf.keras.utils.image_dataset_from_directory(
    "dataSet/train",
    validation_split=0.2,
    subset="validation",
    seed=529,
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=128,
)

#Load test Dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    'dataSet/test',
    image_size=(48,48),
    color_mode='grayscale',
    batch_size=128,
)

# Verify whether the data has been loaded successfully
class_name = train_ds.class_names
print(class_name)


data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.1),
    # Normalization combined with random brightness adjustment improves the model's robustness
    layers.RandomBrightness(factor=0.1),
    layers.Rescaling(1./255),

])

# Convert labels to one-hot
train_ds = train_ds.map(data_process.to_Hot)
validate_ds = validate_ds.map(data_process.to_Hot)
test_ds = test_ds.map(data_process.to_Hot)

test_normalize = keras.Sequential([
    layers.Rescaling(1./255)
])

# Show example augmented images
data_process.show_augmentation(train_ds,data_augmentation)

# Apply normalization and augmentation
validate_ds = validate_ds.map(lambda x,y: (test_normalize(x), y))
test_ds = test_ds.map(lambda x,y: (test_normalize(x), y))
train_ds = train_ds.map(lambda x,y:(data_augmentation(x),y))

# Prefetch to improve performance
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
validate_ds = validate_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Start to build model
model = Sequential()

# Convolutional Block 1
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Convolutional Block 2
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Fully Connected Block
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


# Output the model details
model.summary()


# Start train
model.compile(
    optimizer='adam',
    # loss function
    loss=focal_loss.multi_category_focal_loss1([1.15, 1.5, 1.23, 1.0, 1.1, 1.12, 1.22]),
    metrics=['accuracy'],

)

procedure = model.fit(
    train_ds,
    validation_data=validate_ds,
    epochs=100,
    callbacks=[
        # Stop training if val loss doesn't improve for 10 epochs
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        # Save the best model based on val loss
    tf.keras.callbacks.ModelCheckpoint('model/augment_combine/augment_disgust.keras', save_best_only=True)
],
)


test_loss, test_acc = model.evaluate(test_ds)
plt.plot(procedure.history['accuracy'],label = 'train accuracy')
plt.plot(procedure.history['val_accuracy'], label='validate accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train and Validate Accuracy')
plt.show()

print(f"test_loss:{test_loss}\n")
print(f"test_acc:{test_acc}")
