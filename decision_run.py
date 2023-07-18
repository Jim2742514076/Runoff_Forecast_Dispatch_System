# -*-coding = utf-8 -*-
# @Time : 2023/7/18 9:49
# @Author : 万锦
# @File : decision_run.py
# @Softwore : PyCharm

from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
import time

from ui.decision import Ui_Form

# ui,_ = loadUiType("./ui/predict.ui")

class Form_decision(QWidget,Ui_Form):
    def __init__(self):
        super(Form_decision, self).__init__()
        self.setupUi(self)


def main():
    app = QApplication(sys.argv)
    mainwindow = Form_decision()
    mainwindow.setWindowTitle("方案决策")
    mainwindow.setWindowIcon(QIcon("./icons/Decision_white.svg"))
    mainwindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()