# -*-coding = utf-8 -*-
# @Time : 2023/7/18 19:33
# @Author : 万锦
# @File : main.py
# @Softwore : PyCharm

from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtWebEngineWidgets import QWebEngineView
import sys
import time

from ui.dispatch import Ui_Form

ui,_ = loadUiType("./ui/html_main.ui")

class Form_main(QWidget,ui):
    def __init__(self):
        super(Form_main, self).__init__()
        self.setupUi(self)
        self.setObjectName("main_form")

        self.web =QWebEngineView()
        if self.verticalLayout_2.count() > 0:
            self.verticalLayout_2.removeItem(self.verticalLayout_2.itemAt(0))
        self.web.load(QUrl("file:///" + "./html/主页/index.html"))
        self.verticalLayout_2.addWidget(self.web)




def main():
    app = QApplication(sys.argv)
    mainwindow = Form_main()
    mainwindow.setWindowTitle("主页")
    mainwindow.setWindowIcon(QIcon("./icons/Dispatch_white.svg"))
    mainwindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()