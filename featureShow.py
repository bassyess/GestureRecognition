from keras.models import load_model
from keras import backend as K
from keras.models import Model
import cv2
import numpy as np
import matplotlib.pyplot as plt

modelname = 'VGG16colormodel.h5'
filename = r'H:\ASLdatasetTest\fingerspelling5color\A\a\color_0_0002.png'

model = load_model(modelname)
model.summary()
# layer_1 = K.function([model.layers[0].input], [model.layers[1].output])
layer_output = Model(model.input, model.get_layer('block1_pool').output)
img = cv2.imread(filename)
img2 = cv2.resize(img,(48,48))
arr = np.asarray(img2,dtype="float32")
arr4v = arr[np.newaxis]
f1 = layer_output.predict([arr4v])
for _ in range(32):
            show_img = f1[:, :, :, _]
            show_img.shape = [24, 24]
            plt.subplot(4, 8, _ + 1)
            plt.imshow(show_img, cmap='gray')
            plt.axis('off')
plt.show()