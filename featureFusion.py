import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from keras.models import load_model
from keras.models import Model,Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix
import pickle as pkl
import time


modelname1 = 'VGG16colormodel.h5'
modelname2 = 'VGG16depthmodel.h5'

def get_subdataset(train_data,train_label):
    train_data /=255
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
    return xTrain,xTest,yTrain,yTest

def svc(traindata,trainlabel,testdata,testlabel):
    print("Start training SVM...")
    svcClf = SVC(C=1.0,kernel="linear",cache_size=3000)
    svcClf.fit(traindata,trainlabel)

    pred_testlabel = svcClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("cnn-svm Accuracy:",accuracy)
    print(confusion_matrix(testlabel,pred_testlabel))

read_color = open('colorTest.pkl', 'rb')
color_data = pkl.load(read_color)
color_label = pkl.load(read_color)
read_color.close()
read_depth = open('depthTest.pkl', 'rb')
depth_data = pkl.load(read_depth)
depth_label = pkl.load(read_depth)
read_depth.close()

ctraindata,ctestdata,ctrainlabel,ctestlabel = get_subdataset(color_data,color_label)
dtraindata,dtestdata,dtrainlabel,dtestlabel = get_subdataset(depth_data,depth_label)

c_train_ohe = np_utils.to_categorical(ctrainlabel, 10)
c_test_ohe = np_utils.to_categorical(ctestlabel, 10)
d_train_ohe = np_utils.to_categorical(dtrainlabel, 10)
d_test_ohe = np_utils.to_categorical(dtestlabel, 10)

# time0 = time.time()
# model_color = vgg16Model.vgg16model(ctraindata,ctestdata,c_train_ohe,c_test_ohe)
# time1 = time.time()
# print("生成彩色图模型总共耗时:", str(time1-time0), "s")
# model_depth = vgg16Model.vgg16model(dtraindata,dtestdata,d_train_ohe,d_test_ohe)
# time2 = time.time()
# print("生成深度图模型总共耗时:", str(time2-time1), "s")

model_color = load_model(modelname1)
model_depth = load_model(modelname2)

color_output = Model(model_color.input, model_color.get_layer('flatten').output)
depth_output = Model(model_depth.input, model_depth.get_layer('flatten').output)

colorscores = model_color.evaluate(ctestdata, c_test_ohe, verbose=0)
print("彩色图VGG16的识别结果：",colorscores)
predictioncolor = []
for i in range(ctestdata.shape[0]):
    o = model_color.predict(ctestdata[i][np.newaxis] )
    predictioncolor.append(np.argmax(o))
print (confusion_matrix(ctestlabel,predictioncolor))
depthscores = model_depth.evaluate(dtestdata, d_test_ohe, verbose=0)
print("深度图VGG16的识别结果：",depthscores)
predictiondepth = []
for i in range(dtestdata.shape[0]):
    o = model_depth.predict(dtestdata[i][np.newaxis] )
    predictiondepth.append(np.argmax(o))
print (confusion_matrix(dtestlabel,predictiondepth))


time3 = time.time()
color_train_feature = color_output.predict(ctraindata)
color_test_feature = color_output.predict(ctestdata)
svc(color_train_feature,ctrainlabel,color_test_feature,ctestlabel)
time4 = time.time()
print("彩色图CNN-SVM模型总共耗时:", str(time4-time3), "s")
depth_train_feature = depth_output.predict(dtraindata)
depth_test_feature = depth_output.predict(dtestdata)
svc(depth_train_feature,dtrainlabel,depth_test_feature,dtestlabel)
time5 = time.time()
print("深度图CNN-SVM模型总共耗时:", str(time5-time4), "s")

final_train_feature = color_train_feature*0.6+depth_train_feature*0.4
final_test_feature = color_test_feature*0.6+depth_test_feature*0.4
# final_train_feature = np.concatenate((color_train_feature,depth_train_feature),axis = 0)
# final_test_feature = np.concatenate((color_test_feature,depth_test_feature),axis = 0)
final_train_ohe = np.concatenate((c_train_ohe,d_train_ohe),axis=0)
final_test_ohe = np.concatenate((c_test_ohe,d_test_ohe),axis=0)
final_train_label = np.concatenate((ctrainlabel,dtrainlabel),axis = 0)
final_test_label = np.concatenate((ctestlabel,dtestlabel),axis = 0)
time6 = time.time()
# svc(final_train_feature,final_train_label,final_test_feature,final_test_label)
svc(final_train_feature,ctrainlabel,final_test_feature,ctestlabel)
time7 = time.time()
print("融合的CNN-SVM模型总共耗时:", str(time7-time6), "s")


final_model = Sequential([Dense(32, input_dim=512), Activation('relu'),  Dense(10), Activation('softmax'),])
sgd = SGD(lr=5e-3, decay=0.0005)
final_model.compile(loss="categorical_crossentropy", optimizer="sgd",metrics=["accuracy"])
# final_model.fit(final_train_feature, final_train_ohe,epochs=10, batch_size=128)
# scores = final_model.evaluate(final_test_feature, final_test_ohe, verbose=0)
final_model.fit(final_train_feature, c_train_ohe,epochs=10, batch_size=128)
scores = final_model.evaluate(final_test_feature, c_test_ohe, verbose=0)
print('score=', scores)
predictions = []
for i in range(final_test_feature.shape[0]):
    o = final_model.predict(final_test_feature[i][np.newaxis] )
    predictions.append(np.argmax(o))
print (confusion_matrix(final_test_label,predictions))
# print (confusion_matrix(ctestlabel,predictions))