# -*-coding = utf-8 -*-
# @Time : 2023/7/18 19:33
# @Author : 万锦
# @File : main.py
# @Softwore : PyCharm

import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout

from qfluentwidgets import (PushButton, TeachingTip, TeachingTipTailPosition, InfoBarIcon, setTheme, Theme,
                            TeachingTipView, FlyoutViewBase, BodyLabel, PrimaryPushButton, PopupTeachingTip)


def showTopTip():
    position = TeachingTipTailPosition.BOTTOM
    view = TeachingTipView(
        icon=None,
        title='Lesson 5',
        content="最短的捷径就是绕远路，绕远路才是我的最短捷径。",
        # image='resource/Gyro.jpg',
        # image='resource/boqi.gif',
        isClosable=True,
        tailPosition=position,
    )

    # add widget to view
    button = PushButton('Action')
    button.setFixedWidth(120)
    view.addWidget(button, align=Qt.AlignRight)



showTopTip()