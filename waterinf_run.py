# -*-coding = utf-8 -*-
# @Time : 2023/7/18 9:48
# @Author : 万锦
# @File : waterinf_run.py
# @Softwore : PyCharm

from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
import time

from ui.waterinf import Ui_Form

ui,_ = loadUiType("./ui/waterinf.ui")

class Form_waterinf(QWidget,Ui_Form):
    def __init__(self):
        super(Form_waterinf, self).__init__()
        self.setupUi(self)


def main():
    app = QApplication(sys.argv)
    mainwindow = Form_waterinf()
    mainwindow.setWindowTitle("水情分析")
    mainwindow.setWindowIcon(QIcon("./icons/Waterinf_white.svg"))
    mainwindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()