# -*-coding = utf-8 -*-
# @Time : 2023/7/17 20:42
# @Author : 万锦
# @File : run.py
# @Softwore : PyCharm

import sys
import time
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QIcon, QDesktopServices,QPixmap
from PyQt5.QtWidgets import QApplication, QWidget,QStatusBar,QVBoxLayout,QLabel,QMainWindow,QPushButton,QLineEdit
from qfluentwidgets import (NavigationItemPosition, MessageBox, setTheme, Theme,
                            NavigationAvatarWidget,  SplitFluentWindow, FluentTranslator)
from qfluentwidgets import FluentIcon as FIF
from enum import Enum
from qfluentwidgets import getIconColor, Theme, FluentIconBase

from utils.tools import SplashScreen,MyFluentIcon

from waterinf_run import Form_waterinf
from predict_run import Form_predict
from dispatch_run import Form_dispatch
from decision_run import Form_decision

# class Mywidow(QMainWindow):
#     def __init__(self):
#         super(Mywidow, self).__init__()
#         self.initUI()
#
#
#     def initUI(self):
#         self.resize(300, 150)
#         self.lb = QLabel("文件数量", self)
#         self.lb.move(20, 40)
#         self.bt1 = QPushButton('开始', self)
#         self.bt1.move(20, 80)
#         self.edit = QLineEdit('100000', self)
#         self.edit.move(100, 40)
#         self.show()




class Window(SplitFluentWindow):
    def __init__(self):
        super(SplitFluentWindow,self).__init__()
        self.create_sub()
        self.initNavigation()
    def create_sub(self):
        # create sub interface
        self.waterinf = Form_waterinf()
        self.predict = Form_predict()
        self.dispatch = Form_dispatch()
        self.decision = Form_decision()
        # self.test = Mywidow()
    def initNavigation(self):
        self.addSubInterface(self.waterinf, MyFluentIcon.Waterinf, '水情分析')
        self.addSubInterface(self.predict, MyFluentIcon.Predict,"来水预报")
        self.addSubInterface(self.dispatch,MyFluentIcon.Dispatch,"水量调度")
        self.addSubInterface(self.decision, MyFluentIcon.Decision,"方案决策")
        # self.addSubInterface(self.test,MyFluentIcon.Db,"测试页面")




if __name__ == '__main__':

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    # setTheme(Theme.DARK)

    app = QApplication(sys.argv)

    splash = SplashScreen()
    time.sleep(1)
    splash.close()



    # install translator
    translator = FluentTranslator()
    app.installTranslator(translator)

    w = Window()
    w.show()
    app.exec_()