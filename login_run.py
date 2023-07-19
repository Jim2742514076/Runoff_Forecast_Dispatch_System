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
import configparser

class Form_login(QWidget,Ui_Form):
    def __init__(self):
        super(Form_login, self).__init__()
        self.setupUi(self)
        self.setObjectName("login")
        self.inin_ui()
        self.handle_button()
        self.config_load()


    def handle_button(self):
        self.pushButton.clicked.connect(self.login)
        self.pushButton_2.clicked.connect(self.remember_password)

    def config_load(self):
        # 加载配置文件
        self.config = configparser.ConfigParser()
        self.config.read("./utils/config.ini")

        # 如果配置文件中有保存的用户名和密码，自动填充
        if self.config.has_option("Login", "username") and self.config.has_option("Login", "password"):
            username = self.config.get("Login", "username")
            password = self.config.get("Login", "password")
            self.lineEdit_3.setText(username)
            self.lineEdit_4.setText(password)
            self.checkBox.setChecked(True)

    #记住密码
    def remember_password(self):
        if self.lineEdit_3.text():
            username = self.lineEdit_3.text()
            conn,cursor = get_conn()
            sql = "SELECT * FROM users where username=%s"
            cursor.execute(sql,(username))
            data = cursor.fetchone()
            if data:
                self.lineEdit_4.setText(data[2])
                self.showTopTip_success()
            else:
                self.showTopTip_remember_error()


    def showTopTip_remember_error(self):
        position = TeachingTipTailPosition.BOTTOM
        view = TeachingTipView(
            icon=None,
            title='系统无法检索操作员',
            content="人类把最精密的保密系统，都用在了自我毁灭上。",
            image='./icons/user_error.jpeg',
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
    def showTopTip_success(self):
        position = TeachingTipTailPosition.BOTTOM
        view = TeachingTipView(
            icon=None,
            title="点火成功",
            content="你觉得不公平吗？危难当前，唯有责任。",
            image='./icons/earth_successful.png',
            isClosable=True,
            tailPosition=position,
        )

        # add widget to view
        button = PushButton('了解')
        button.setFixedWidth(120)
        view.addWidget(button, align=Qt.AlignRight)


    def showTopTip(self):
        position = TeachingTipTailPosition.BOTTOM
        view = TeachingTipView(
            icon=None,
            title='指令错误',
            content="生存的最大障碍，从不是弱小，而是傲慢。",
            image='./icons/login_error.jpeg',
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
    def showTopTip_success(self):
        position = TeachingTipTailPosition.BOTTOM
        view = TeachingTipView(
            icon=None,
            title="点火成功",
            content="你觉得不公平吗？危难当前，唯有责任。",
            image='./icons/earth_successful.png',
            # image='resource/boqi.gif',
            isClosable=True,
            tailPosition=position,
        )

        # add widget to view
        button = PushButton('了解')
        button.setFixedWidth(120)
        view.addWidget(button, align=Qt.AlignRight)

        w = TeachingTip.make(
            target=self.pushButton_2,
            view=view,
            duration=-1,
            tailPosition=position,
            parent=self
        )
        view.closed.connect(w.close)

    def showTopTip_mima(self):
        position = TeachingTipTailPosition.BOTTOM
        view = TeachingTipView(
            icon=None,
            title="密码错误",
            content="生存的最大障碍，从不是弱小，而是傲慢。",
            image='./icons/login_error.jpeg',
            # image='resource/boqi.gif',
            isClosable=True,
            tailPosition=position,
        )

        # add widget to view
        button = PushButton('了解')
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

        remember_password = self.checkBox.isChecked()
        # 保存用户名和密码到配置文件
        if remember_password:
            self.config["Login"] = {"username": username, "password": password}
            with open("./utils/config.ini", "w") as configfile:
                self.config.write(configfile)

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
                self.showTopTip_mima()
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

    splash = SplashScreen()
    splash.show()
    time.sleep(3)
    splash.close()

    mainwindow = Form_login()
    mainwindow.setWindowTitle("登录")
    mainwindow.setWindowIcon(QIcon("./icons/UEG.png"))
    mainwindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()