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
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import SGD
import focal_loss
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import optimizers
from tensorflow.keras.applications import MobileNetV2

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
    batch_size=32,
)

# Load validation dataset
validate_ds = tf.keras.utils.image_dataset_from_directory(
    "dataSet/train",
    validation_split=0.2,
    subset="validation",
    seed=529,
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
)

#Load test Dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    'dataSet/test',
    image_size=(48,48),
    color_mode='grayscale',
    batch_size=32,
)

# Verify whether the data has been loaded successfully
class_name = train_ds.class_names
print(class_name)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1),
    layers.Rescaling(1./255),
])

# - convert grayscale → RGB for MobileNetV2
train_ds = train_ds.map(data_process.convert_VGG)
test_ds = test_ds.map(data_process.convert_VGG)
validate_ds = validate_ds.map(data_process.convert_VGG)

# convert label to on-hot
train_ds = train_ds.map(data_process.to_Hot)
validate_ds = validate_ds.map(data_process.to_Hot)
test_ds = test_ds.map(data_process.to_Hot)

test_normalize = keras.Sequential([
    layers.Rescaling(1./255)
])

data_process.augmentation(train_ds,data_augmentation)

# Normalization layer for validation/test (no augmentation)
validate_ds = validate_ds.map(lambda x,y: (test_normalize(x), y))
test_ds = test_ds.map(lambda x,y: (test_normalize(x), y))
# Augment the train dataset
train_ds = train_ds.map(lambda x,y: (data_augmentation(x), y))

train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
validate_ds = validate_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# import MOBIL
mobilenet = MobileNetV2(weights='imagenet',
                        include_top=False,
                        input_shape=(48, 48, 3))

# Freeze all convolutional layers
for layer in mobilenet.layers:
    layer.trainable = False

# Add custom classification head
head = mobilenet.output
head = Dense(256, activation='relu')(head)
head = Dropout(0.5)(head)
head = GlobalAveragePooling2D()(head)
head = Dense(7, activation='softmax')(head)

# Build final model

model = Model(inputs=mobilenet.input, outputs=head)


print(model.summary())

optims = [optimizers.Adam(learning_rate = 0.0001,
                          beta_1 = 0.9, beta_2 = 0.999),]
model.compile(loss= focal_loss.multi_category_focal_loss1([1,1.6,1,1,1,1,1]), metrics=['accuracy'], optimizer=optims[0])


procedure = model.fit(
    train_ds,
    validation_data=validate_ds,
    batch_size=32,
    epochs=100,
    callbacks = [
        # Save the best model (based on validation accuracy)
        ModelCheckpoint(
            filepath="best_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        # Stop early if accuracy does not improve
        EarlyStopping(monitor = 'val_accuracy',
                       min_delta = 0.00005,
                       patience = 11,
                       verbose = 1,
                       restore_best_weights = True,),
        
         # Reduce learning rate if validation accuracy plateaus
        ReduceLROnPlateau(monitor = 'val_accuracy',
                         factor = 0.5,
                         patience = 7,
                         min_lr = 1e-7,
                         verbose = 1,)
    ]
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

