import numpy as np
import time
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import pickle as pkl
from keras.models import load_model
# import cnnModel
import vgg16Model

modelname1 = 'VGG16colormodel.h5'
modelname2 = 'VGG16depthmodel.h5'

def get_subdataset(train_data,train_label):
    #train_data,train_label = load_data(filepath)
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

def findnumber(preds):
    num = 0
    prob = 0
    for i in range(0, len(preds)):
        if preds[i] > prob:
            prob = preds[i]
            num = i
        else:
            pass
    return num


read_color = open('colorTest.pkl', 'rb')
color_data = pkl.load(read_color)
color_label = pkl.load(read_color)
read_color.close()
read_depth = open('depthTest.pkl', 'rb')
depth_data = pkl.load(read_depth)
depth_label = pkl.load(read_depth)
read_depth.close()

time0 = time.time()
ctraindata,ctestdata,ctrainlabel,ctestlabel = get_subdataset(color_data,color_label)
time1 = time.time()
print("生成彩色图数据集总共耗时:", str(time1-time0), "s")
dtraindata,dtestdata,dtrainlabel,dtestlabel = get_subdataset(depth_data,depth_label)
time2 = time.time()
print("生成深度图数据集总共耗时:", str(time2-time1), "s")


# # 把标签用One Hot编码重新表示
c_train_ohe = np_utils.to_categorical(ctrainlabel, 10)
c_test_ohe = np_utils.to_categorical(ctestlabel, 10)
d_train_ohe = np_utils.to_categorical(dtrainlabel, 10)
d_test_ohe = np_utils.to_categorical(dtestlabel, 10)

# time3 = time.time()
# model_color = vgg16Model.vgg16model(ctraindata,ctestdata,c_train_ohe,c_test_ohe,20)
# time4 = time.time()
# print("生成彩色图模型总共耗时:", str(time4-time3), "s")
# model_depth = vgg16Model.vgg16model(dtraindata,dtestdata,d_train_ohe,d_test_ohe,40)
# time5 = time.time()
# print("生成深度图模型总共耗时:", str(time5-time4), "s")
model_color = load_model(modelname1)
model_depth = load_model(modelname2)

colorscores = model_color.evaluate(ctestdata, c_test_ohe, verbose=0)
print("彩色图VGG16的识别结果：",colorscores)
depthscores = model_depth.evaluate(dtestdata, d_test_ohe, verbose=0)
print("深度图VGG16的识别结果：",depthscores)

#生成结果融合
nrows = min(len(ctestdata), len(dtestdata))
count1,count2,count3,count4,count5 = 0,0,0,0,0
acu1,acu2,acu3,acu4,acu5 = 0.0,0.0,0.0,0.0,0.0
# Wa = [0.1069,0.1068,0.1053,0.0896,0.0930,0.1053,0.1030,0.0973,0.1030,0.0898]
# Wb = [0.0848,0.1029,0.1036,0.1015,0.0857,0.1057,0.1032,0.1087,0.1027,0.1011]
Wa = [0.58,0.50,0.51,0.51,0.51,0.51,0.49,0.50,0.51,0.51]
Wb = [0.42,0.50,0.49,0.49,0.49,0.49,0.51,0.50,0.49,0.49]
# Wa = [0.996,0.996,0.986,0.996,0.986,0.998,0.942,0.992,0.988,0.988]
# Wb = [0.772,0.98,0.938,0.956,0.594,0.972,0.968,0.99,0.962,0.942]
for k in range(0, nrows):
    colorarr4v = ctestdata[k][np.newaxis]
    deptharr4v = dtestdata[k][np.newaxis]
    predA = model_color.predict(colorarr4v)
    predB = model_depth.predict(deptharr4v)
    m = min(len(predA[0]), len(predB[0]))
    #最大值融合
    predmax = []
    for i in range(0, m):
        if predA[0][i] > predB[0][i]:
            predmax.append(predA[0][i])
        else:
            predmax.append(predB[0][i])
    predmax = np.array(predmax)
    num1 = findnumber(predmax)
    if (ctestlabel[k] == num1) and (dtestlabel[k] == num1):
        count1 = count1 + 1
    #平均值融合
    predave = []
    for i in range(0, m):
        predave.append(predA[0][i]*0.5 + predB[0][i]*0.5)
    predave = np.array(predave)
    num2 = findnumber(predave)
    if (ctestlabel[k] == num2) and (dtestlabel[k] == num2):
        count2 = count2 + 1
    #乘性融合
    predmul = []
    for i in range(0, m):
        predmul.append(predA[0][i] * predB[0][i])
    predmul = np.array(predmul)
    num3 = findnumber(predmul)
    if (ctestlabel[k] == num3) and (dtestlabel[k] == num3):
        count3 = count3 + 1
    # 改进的加权融合
    wn = min(len(Wa), len(Wb))
    A = []
    for i in range(0, wn):
        temA = [Wa[i] * x for x in predA[0]]
        temB = [Wb[i] * y for y in predB[0]]
        n = min(len(temA), len(temB))
        temC = []
        for j in range(0, n):
            temC.append(temA[j] + temB[j])
        A.append(temC)
    A = np.array(A)
    score = []
    # sum = []
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[1]):
            one = A[i][0]
            two = A[i][0]
            if A[i][j] > one:
                two = one
                one = A[i][j]
            elif A[i][j] > two:
                two = A[i][j]
            else:
                pass
        if (A[i][i] == one):
            score.append(A[i][i] - two)
        else:
            score.append(A[i][i] - one)
    score = np.array(score)
    # num4 = findnumber(sum)
    # if (ctestlabel[k] == num4) and (dtestlabel[k] == num4):
    #     count4 = count4 + 1
    num5 = findnumber(score)
    if (ctestlabel[k] == num5) and (dtestlabel[k] == num5):
        count5 = count5 + 1



acu1 = count1/nrows
print("最大值融合的识别概率：",+acu1)
acu2 = count2/nrows
print("平均值融合的识别概率：",+acu2)
acu3 = count3/nrows
print("乘性融合的识别概率：",+acu3)
# acu4 = count4/nrows
# print("加权融合的识别概率：",+acu4)
acu5 = count5/nrows
print("改进的加权融合的识别概率：",+acu5)