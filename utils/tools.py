# -*-coding = utf-8 -*-
# @Time : 2023/7/18 10:18
# @Author : 万锦
# @File : icons_tools.py
# @Softwore : PyCharm

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QVBoxLayout,QSplashScreen,QLabel
from qfluentwidgets import getIconColor, Theme, FluentIconBase
from enum import Enum

#自定义图标
class MyFluentIcon(FluentIconBase, Enum):
    """ Custom icons """
    AliPay = "AliPay"
    Db = "Db"
    Decision = "Decision"
    Dispatch = "Dispatch"
    Excel = "Excel"
    Predict = "Predict"
    Waterinf = "Waterinf"
    System = "System"

    def path(self, theme=Theme.AUTO):
        return f'./icons/{self.value}_{getIconColor(theme)}.svg'


#数据库配置





#项目启动
class SplashScreen(QSplashScreen):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('加载中...')
        self.setPixmap(QPixmap('./icons/550w.bmp'))  # 设置加载页面的图片

        # 创建布局和标签
        layout = QVBoxLayout()
        label = QLabel('')
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        # 将布局设置给加载页面
        self.setLayout(layout)
        self.show()

