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
from qfluentwidgets import (PushButton, Flyout, InfoBarIcon, FlyoutView, FlyoutViewBase,
                            BodyLabel, setFont, PrimaryPushButton, FlyoutAnimationType)
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
        self.create_nag()

        # splash = SplashScreen()
        # time.sleep(5)
        # splash.close()

    def inin_title(self):
        # 设置窗体标题
        self.setWindowTitle("径流预报调度系统")
        self.setWindowIcon(QIcon("./icons/System_black.svg"))
        self.theme_change_control = 1
        # 设置窗口大小
        self.resize(1000, 650)
        # 居中显示
        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)


    def create_nag(self):
        self.navigationInterface.addWidget(
            routeKey='avatar',
            widget=NavigationAvatarWidget('Jim', './icons/user.jpg'),
            onClick=self.inf_user,
            position=NavigationItemPosition.BOTTOM,
        )
        self.navigationInterface.addWidget(
            routeKey='settingInterface',
            widget=NavigationAvatarWidget('设置', "./icons/theme.svg"),
            onClick=self.change_theme,
            position=NavigationItemPosition.BOTTOM,
        )

        self.navigationInterface.setExpandWidth(280)

    def inf_user(self):

        w = MessageBox(
            '看一眼就爆炸🥰',
            '本项目仅供学习使用，不可用于任何形式的商业用途，如果一定要商用请仔细阅读源码，排除代码里留的后门，因非法商用造成的一切后果与作者无关',
            self
        )
        w.yesButton.setText('原地爆炸')
        w.cancelButton.setText('下次一定')

        if w.exec():
            QDesktopServices.openUrl(QUrl("https://github.com/Jim2742514076/Runoff_Forecast_Dispatch_System"))

    def change_theme(self):
        if self.theme_change_control % 2 == 1:
            self.theme_change_control +=1
            setTheme(Theme.DARK)
        else:
            self.theme_change_control +=1
            setTheme(Theme.LIGHT)

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