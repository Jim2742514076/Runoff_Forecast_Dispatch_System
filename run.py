# -*-coding = utf-8 -*-
# @Time : 2023/7/17 20:42
# @Author : 万锦
# @File : run.py
# @Softwore : PyCharm

import sys
import time
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QIcon, QDesktopServices,QPixmap,QFont
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







class Window(SplitFluentWindow):
    def __init__(self):
        super(SplitFluentWindow,self).__init__()
        self.create_sub()
        self.initNavigation()
        self.inin_title()

        # splash = SplashScreen()
        # time.sleep(5)
        # splash.close()

    def inin_title(self):
        # 设置窗体标题
        self.setWindowTitle("径流预报调度系统")
        self.setWindowIcon(QIcon("./icons/System_black.svg"))
        # 设置窗口大小
        self.resize(1000, 650)
        # 居中显示
        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)




    def create_sub(self):
        # create sub interface
        self.waterinf = Form_waterinf()
        self.predict = Form_predict()
        self.dispatch = Form_dispatch()
        self.decision = Form_decision()


    def initNavigation(self):
        self.addSubInterface(self.waterinf, MyFluentIcon.Waterinf, '水情分析')
        self.addSubInterface(self.predict, MyFluentIcon.Predict,"来水预报")
        self.addSubInterface(self.dispatch,MyFluentIcon.Dispatch,"水量调度")
        self.addSubInterface(self.decision, MyFluentIcon.Decision,"方案决策")





def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    # setTheme(Theme.DARK)

    app = QApplication(sys.argv)

    # #设置开场
    # splash = SplashScreen()
    # time.sleep(3)
    # splash.close()

    # install translator
    translator = FluentTranslator()
    app.installTranslator(translator)

    w = Window()
    # w.setWindowTitle("")

    w.show()
    app.exec_()

if __name__ == '__main__':
    main()