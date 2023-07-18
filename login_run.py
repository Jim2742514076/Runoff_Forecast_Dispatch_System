# -*-coding = utf-8 -*-
# @Time : 2023/7/18 17:55
# @Author : 万锦
# @File : login_run.py
# @Softwore : PyCharm

from PyQt5.QtGui import *
from PyQt5.QtWidgets import QWidget,QApplication
from PyQt5.QtCore import *
import sys
import time
from utils.db_tools import get_conn,close_conn
from utils.tools import MyFluentIcon
from utils.tools import SplashScreen,MyFluentIcon
from ui.login import Ui_Form
from run import Window
from qfluentwidgets import (PushButton, TeachingTip, TeachingTipTailPosition, InfoBarIcon, setTheme, Theme,
                            TeachingTipView, FlyoutViewBase, BodyLabel, PrimaryPushButton, PopupTeachingTip)


class Form_login(QWidget,Ui_Form):
    def __init__(self):
        super(Form_login, self).__init__()
        self.setupUi(self)
        self.setObjectName("login")
        self.inin_ui()
        self.handle_button()


    def handle_button(self):
        self.pushButton.clicked.connect(self.login)

    def showTopTip(self):
        position = TeachingTipTailPosition.BOTTOM
        view = TeachingTipView(
            icon=None,
            title='指令错误',
            content="最短的捷径就是绕远路，绕远路才是我的最短捷径。",
            image='./icons/550w.bmp',
            # image='resource/boqi.gif',
            isClosable=True,
            tailPosition=position,
        )

        # add widget to view
        button = PushButton('Action')
        button.setFixedWidth(120)
        view.addWidget(button, align=Qt.AlignRight)

        w = TeachingTip.make(
            target=self.pushButton,
            view=view,
            duration=-1,
            tailPosition=position,
            parent=self
        )
        view.closed.connect(w.close)

    def login(self):
        username = self.lineEdit_3.text()
        password = self.lineEdit_4.text()

        conn,cursor = get_conn()
        sql = "SELECT username,password FROM users WHERE username=%s"
        cursor.execute(sql,(username,))
        inf = cursor.fetchone()
        if inf:
            if inf[1] == password:
                close_conn(conn,cursor)
                self.close()
                self.open_main_run()
            else:
                self.showTopTip()
                close_conn(conn, cursor)
        else:
            self.showTopTip()
            close_conn(conn, cursor)



    def open_main_run(self):
        window = Window()
        window.show()




    def inin_ui(self):
        self.setObjectName("dispatch_form")
        self.label.setPixmap(QPixmap("./icons/550w.bmp"))
        self.label_2.setPixmap(QPixmap("./icons/UEG.png"))
        self.setWindowTitle("登录")
        self.setWindowIcon(QIcon("./icons/UEG.png"))
        #设置只有关闭按键
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        #设置窗口大小
        self.resize(1100, 600)
        #居中显示
        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)

    # splash = SplashScreen()
    # splash.show()
    # time.sleep(3)
    # splash.close()

    mainwindow = Form_login()
    mainwindow.setWindowTitle("登录")
    mainwindow.setWindowIcon(QIcon("./icons/UEG.png"))
    mainwindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()