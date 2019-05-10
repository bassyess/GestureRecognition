from keras.layers import Dense, Dropout, Flatten,Conv2D,Input,MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle as pkl
from sklearn.metrics import confusion_matrix, classification_report
import cv2
import os

# def load_data(rootDir):
#     i = 0
#     data = np.empty((1,224,224,3),dtype="float32")   # 初始化标签和数据
#     label = np.empty((1,), dtype="uint8")
#     for root,dirs,files in os.walk(rootDir):
#         for file in files:
#             imgadd = os.path.join(root,file)            # 获取图片路径
#             print(imgadd)
#             img = cv2.imread(imgadd)                    # 读取图片
#             img_224 = cv2.resize(img,(224,224))         # 对图片进行剪裁
#             arr = np.asarray(img_224,dtype="float32")  # 将图片转化为三维数组
#             arr4v = arr[np.newaxis]                     # 将表示图片的三维数组变成四维数组
#             if i==0 :                                   # 第一次时，将三位数组添加到第一个的data后三维
#                 data[0,:,:,:] = arr
#                 label[0] = int(imgadd.split('/')[-1].split('_')[1])   # 获取图片名称中的标签
#             else :                                      # 第二次时，将四维数组直接添加到data
#                 data = np.append(data,arr4v,axis=0)
#                 newlabel = np.array([int(imgadd.split('/')[-1].split('_')[1])])   # 获取图片名称中的标签
#                 label = np.append(label,newlabel)
#             i = i+1
#     print(data.shape, label.shape)
#     return data,label

# trainPath = r"/home/bigdata/gaoxinkai/bisheCode/fingerspelling5depth"
time1 = time.time()
# train_data, train_label = load_data(trainPath)   # 获取数据集数据
read_file = open('colorTest.pkl', 'rb')
train_data = pkl.load(read_file)
train_label = pkl.load(read_file)
read_file.close()

#分层抽样，按标签抽取
nrows = len(train_data)
xTemp = [train_data[i] for i in range(nrows) if train_label[i] == 0]
yTemp = [train_label[i] for i in range(nrows) if train_label[i] == 0]
xTrain, xTest, yTrain, yTest = train_test_split(xTemp, yTemp, test_size=0.20, random_state=128)
for iLabel in range(1, 10):
    # segregate x and y according to labels
    xTemp = [train_data[i] for i in range(nrows) if train_label[i] == iLabel]
    yTemp = [train_label[i] for i in range(nrows) if train_label[i] == iLabel]
    # form train and test sets on segregated subset of examples
    xTrainTemp, xTestTemp, yTrainTemp, yTestTemp = train_test_split(xTemp, yTemp, test_size=0.20, random_state=128)

    # accumulate
    xTrain = np.append(xTrain, xTrainTemp, axis=0)
    xTest = np.append(xTest, xTestTemp, axis=0)
    yTrain = np.append(yTrain, yTrainTemp, axis=0)
    yTest = np.append(yTest, yTestTemp, axis=0)
print(xTrain.shape, xTest.shape, yTrain.shape, yTest.shape)

# 归一化像素值，控制在0-1范围内
xTrain /= 255
xTest /= 255

# 把目标标签0-24做成One Hot编码形式
def train_y(y):
    y_ohe = np.zeros(10)
    y_ohe[y] = 1
    return y_ohe

# 把标签用One Hot编码重新表示
y_train_ohe = np.array([train_y(yTrain[i]) for i in range(len(yTrain))])
y_test_ohe = np.array([train_y(yTest[i]) for i in range(len(yTest))])

time2 = time.time()
print("总共耗时:", str(time2-time1), "s")


# model_vgg = VGG16(include_top=False, weights="imagenet", input_shape=(48, 48, 3))

inputs = Input(shape=(48, 48, 3))
# Block 1
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Block 3
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# Block 4
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

# Block 5
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

model = Flatten(name="flatten")(x)
model = Dense(1024, activation="relu", name="fc1")(model)
model = Dropout(0.5)(model)
model = Dense(1024, activation="relu", name="fc2")(model)
model = Dropout(0.5)(model)
model = Dense(10, activation="softmax")(model)
model_vgg_asl = Model(inputs, model, name="vgg16")
model_vgg_asl.summary()

sgd = SGD(lr=5e-3, decay=0.0005)
model_vgg_asl.compile(loss="categorical_crossentropy", optimizer="sgd",metrics=["accuracy"])
history = model_vgg_asl.fit(xTrain, y_train_ohe, validation_data=(xTest, y_test_ohe),epochs=5, batch_size=100)
scores = model_vgg_asl.evaluate(xTest, y_test_ohe, verbose=0)
print('score=', scores)

fig,ax = plt.subplots(2,1,figsize=(10,10))
ax[0].plot(history.history['loss'], color='r', label='Training Loss')
ax[0].plot(history.history['val_loss'], color='g', label='Validation Loss')
ax[0].legend(loc='best',shadow=True)
ax[0].grid(True)

ax[1].plot(history.history['acc'], color='r', label='Training Accuracy')
ax[1].plot(history.history['val_acc'], color='g', label='Validation Accuracy')
ax[1].legend(loc='best',shadow=True)
ax[1].grid(True)
fig.show()


# model_vgg_asl.save("gesturemodel3.h5")

# time3 = time.time()
# print("神经网络总共耗时", str(time3-time2), "s")
# predictions = []
# for i in range(xTest.shape[0]):
#     o = model_vgg_asl.predict(xTest[i][np.newaxis] )
#     predictions.append(np.argmax(o))
# print (confusion_matrix(yTest,predictions))
# print (classification_report(yTest,predictions))

