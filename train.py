import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense
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
train_ds = train_ds.map(data_process.to_Hot)
validate_ds = validate_ds.map(data_process.to_Hot)
test_ds = test_ds.map(data_process.to_Hot)

test_normalize = keras.Sequential([
    layers.Rescaling(1./255)
])

data_process.augmentation(train_ds,data_augmentation)

validate_ds = validate_ds.map(lambda x,y: (test_normalize(x), y))
test_ds = test_ds.map(lambda x,y: (test_normalize(x), y))
train_ds = train_ds.map(lambda x,y: (data_augmentation(x), y))

train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
validate_ds = validate_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Start to build model
model = Sequential()
model.add(Conv2D(
    filters=32,
    kernel_size=3,
    padding='same',
    activation='relu',
    input_shape=(48,48,1)
))

model.add(Conv2D(
    filters=32,
    kernel_size=3,
    padding='same',
    activation='relu',
))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(
    filters=64,
    kernel_size=3,
    padding='same',
    activation='relu',
))

model.add(Conv2D(
    filters=64,
    kernel_size=3,
    padding='same',
    activation='relu',
))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(
    filters=128,
    kernel_size=3,
    padding='same',
    activation='relu',
))

model.add(Conv2D(
    filters=128,
    kernel_size=3,
    padding='same',
    activation='relu',
))

model.add(MaxPooling2D(pool_size=2))

model.add(layers.Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(7,activation='softmax'))


# Output the model details
model.summary()


# Start train
model.compile(
    optimizer='adam',
    # loss function
    loss=focal_loss.multi_category_focal_loss1([1.15, 2.3, 1.14, 1.0, 1.1, 1.12, 1.22]),
    metrics=['accuracy'],

)

procedure = model.fit(
    train_ds,
    validation_data=validate_ds,
    epochs=100,
    callbacks=[
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('model/model_second_focal/focal_loss.keras', save_best_only=True)
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

