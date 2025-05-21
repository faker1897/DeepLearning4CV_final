import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
import tensorflow as tf

model = load_model('model/first_CNN_BEST.keras')

test_ds=tf.keras.utils.image_dataset_from_directory(
    'dataSet/test',
    image_size=(48, 48),
    color_mode='grayscale',
    batch_size=128,
)

test_ds=test_ds.map(lambda x,y:(x/255,y))

predict = model.predict(test_ds)
label = np.argmax(predict,axis=1)
label_true = []
for image,label_test in test_ds:
    label_true.extend(label_test.numpy())

label_true = np.array(label_true)

confusion = confusion_matrix(label, label_true)
classification = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

output = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=classification)
output.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()