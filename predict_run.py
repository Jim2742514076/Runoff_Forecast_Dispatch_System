# -*-coding = utf-8 -*-
# @Time : 2023/7/18 9:48
# @Author : 万锦
# @File : predict_run.py
# @Softwore : PyCharm

from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
import time

from ui.predict import Ui_Form

ui,_ = loadUiType("./ui/predict.ui")

class Form_predict(QMainWindow,ui):
    def __init__(self):
        super(Form_predict, self).__init__()
        self.setupUi(self)
        self.setObjectName("predict_form")


def main():
    app = QApplication(sys.argv)
    mainwindow = Form_predict()
    mainwindow.setWindowTitle("来水预测")
    mainwindow.setWindowIcon(QIcon("./icons/Predict_white.svg"))
    mainwindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()