import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.models import load_model
import tensorflow as tf
import data_process
import focal_loss

model = load_model('model/model_first/first_CNN_BEST.keras',custom_objects={
    'multi_category_focal_loss1_fixed': focal_loss
})

test_ds=tf.keras.utils.image_dataset_from_directory(
    'dataSet/test',
    image_size=(48, 48),
    color_mode='grayscale',
    batch_size=128,
    shuffle=False
)

test_ds=test_ds.map(lambda x,y:(x/255,y))
test_ds = test_ds.map(data_process.to_Hot)
predict = model.predict(test_ds)
label_predict = np.argmax(predict, axis=1)
label_true = []
for image,label_test in test_ds:
    label_true.extend(label_test.numpy())

label_true = np.array(label_true)
label_true = np.argmax(label_true, axis=1)
confusion = confusion_matrix(label_true,label_predict )
classification = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

print("Classification Report:\n")
print(classification_report(label_true,label_predict, target_names=classification))

output = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=classification)
output.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()