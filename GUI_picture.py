import sys
import cv2
import numpy as np
from keras.models import load_model
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
# from PyQt5.QtWidgets import QWidget,QApplication,QLabel,QPushButton,QFileDialog
# from PyQt5.QtGui import QPixmap

class Gesture(QWidget):

    def __init__(self):
        super().__init__()
        self.f1 = []
        self.initUI()

    def initUI(self):

        layout = QVBoxLayout()

        self.btn = QPushButton("加载图片")
        self.btn1 = QPushButton("手势识别")
        self.btn.clicked.connect(self.loadFile)
        self.btn1.clicked.connect(self.recognition)
        layout.addWidget(self.btn)
        layout.addWidget(self.btn1)

        self.lb1 = QLabel(self)
        self.lb2 = QLabel(self)
        self.lb3 = QLabel(self)
        self.lb4 = QLabel(self)
        layout.addWidget(self.lb1)
        layout.addWidget(self.lb4)
        layout.addWidget(self.lb2)
        layout.addWidget(self.lb3)

        self.setGeometry(200, 200, 400, 400)
        self.setWindowTitle('Gesture Recognition')
        self.setLayout(layout)
        self.show()

    def loadFile(self):
        print("load--file")
        self.f1 = []
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', 'H:\\', 'Image files(*.jpg *.gif *.png)')
        self.lb1.setPixmap(QPixmap(fname))
        self.f1.append(fname)
        label = str(fname.split('/')[-1].split('_')[1])
        self.lb4.setText("实际对应的手势编号为：" + label)

    def recognition(self):
        print(self.f1[0])
        img = cv2.imread(self.f1[0])
        img_224 = cv2.resize(img.copy(), (48, 48))
        arr = np.asarray(img_224, dtype="float32")
        arr4v = arr[np.newaxis]
        arr4v /= 255
        preds = model.predict(arr4v)
        #print(preds)
        # 输出概率最大的手势
        m = len(preds[0])
        prob = 0  # 最大概率
        num = -1  # 手势编号
        for i in range(0, m):
            if preds[0][i] > prob:
                prob = preds[0][i]
                num = i
            else:
                pass
        print(num)
        num = str(num)
        prob = str(prob)
        self.lb2.setText("识别出来的手势编号为："+num)
        self.lb3.setText("识别为该手势对应的概率："+prob)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    model = load_model('colormodel.h5')
    ges = Gesture()
    sys.exit(app.exec_())