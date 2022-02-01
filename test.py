from tensorflow import keras
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

model = keras.models.load_model('model.h5')
answer = ['With mask','Without mask']

def preprocess_image(path):
    img = load_img(path, target_size = (256, 256))
    a = img_to_array(img)
    a = np.expand_dims(a, axis = 0)
    a /= 255.0
    return a

image_path = '/content/drive/MyDrive/Colab Notebooks/maskdata/test/with_mask/0-with-mask.jpg'
image = preprocess_image(image_path)
predict = model.predict(image)
result = np.argmax(predict)
print(answer[result])
