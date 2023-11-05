# -*- coding: utf-8 -*-
# Author: 万锦
# Email : wanjinhhu@gmail.com
# Time : 2023/11/5 14:53
# File: test.py
# Software: PyCharm


# coding:utf-8
import sys
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout,QProgressDialog,QDialog
from qfluentwidgets import ProgressRing, SpinBox, setTheme, Theme, IndeterminateProgressRing, setFont
import time








class Demo(QDialog):

    def __init__(self):
        super().__init__()
        # setTheme(Theme.DARK)
        # self.setStyleSheet('Demo{background: rgb(32, 32, 32)}')
        self.setWindowTitle("模型训练中")
        self.setWindowIcon(QIcon("./icons/UEG.png"))
        self.vBoxLayout = QVBoxLayout(self)
        self.spinner = IndeterminateProgressRing(self)
        self.vBoxLayout.addWidget(self.spinner, 0, Qt.AlignHCenter)
        self.resize(400, 400)


if __name__ == '__main__':
    # enable dpi scale
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    w = Demo()
    w.show()
    app.exec_()
