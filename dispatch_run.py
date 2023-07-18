# -*-coding = utf-8 -*-
# @Time : 2023/7/18 9:49
# @Author : 万锦
# @File : dispatch_run.py
# @Softwore : PyCharm

from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
import time

from ui.dispatch import Ui_Form

ui,_ = loadUiType("./ui/dispatch.ui")

class Form_dispatch(QMainWindow,ui):
    def __init__(self):
        super(Form_dispatch, self).__init__()
        self.setupUi(self)
        self.setObjectName("dispatch_form")


def main():
    app = QApplication(sys.argv)
    mainwindow = Form_dispatch()
    mainwindow.setWindowTitle("水量调度")
    mainwindow.setWindowIcon(QIcon("./icons/Dispatch_white.svg"))
    mainwindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()