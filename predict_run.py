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

from ui.predict import Ui_MainWindow

ui,_ = loadUiType("./ui/predict.ui")

class Form_predict(QMainWindow,ui):
    def __init__(self):
        super(Form_predict, self).__init__()
        self.setupUi(self)
        self.setObjectName("predict_form")

        self.initialize_combox()

        # 初始化下拉框
        def initialize_combox(self):
            # 从列表中添加下拉选项
            self.ComboBox.addItems([str(_) for _ in range(1, 11)])
            # 设置显示项目
            self.ComboBox.setCurrentIndex(4)





def main():
    app = QApplication(sys.argv)
    mainwindow = Form_predict()
    mainwindow.setWindowTitle("来水预测")
    mainwindow.setWindowIcon(QIcon("./icons/Predict_white.svg"))
    mainwindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()