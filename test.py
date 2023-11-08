# -*- coding: utf-8 -*-
# Author: 万锦
# Email : wanjinhhu@gmail.com
# Time : 2023/11/5 14:53
# File: test.py
# Software: PyCharm


# # coding:utf-8
# import sys
# from PyQt5.QtGui import QIcon
# from PyQt5.QtCore import Qt, QTimer
# from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout,QProgressDialog,QDialog
# from qfluentwidgets import ProgressRing, SpinBox, setTheme, Theme, IndeterminateProgressRing, setFont
# import time
#
#
#
#
#
#
#
#
# class Demo(QDialog):
#
#     def __init__(self):
#         super().__init__()
#         # setTheme(Theme.DARK)
#         # self.setStyleSheet('Demo{background: rgb(32, 32, 32)}')
#         self.setWindowTitle("模型训练中")
#         self.setWindowIcon(QIcon("./icons/UEG.png"))
#         self.vBoxLayout = QVBoxLayout(self)
#         self.spinner = IndeterminateProgressRing(self)
#         self.vBoxLayout.addWidget(self.spinner, 0, Qt.AlignHCenter)
#         self.resize(400, 400)
#
#
# if __name__ == '__main__':
#     # enable dpi scale
#     QApplication.setHighDpiScaleFactorRoundingPolicy(
#         Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
#     QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
#     QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
#
#     app = QApplication(sys.argv)
#     w = Demo()
#     w.show()
#     app.exec_()
# coding:utf-8

# import sys
# from PyQt5.QtCore import Qt, QThread, pyqtSignal
# from PyQt5.QtGui import QPalette
# from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QDialog, QProgressBar
#
# class WorkerThread(QThread):
#     update_progress = pyqtSignal(int)
#
#     def run(self):
#         for i in range(101):
#             self.update_progress.emit(i)
#             self.msleep(100)
#
# class ProgressDialogExample(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.initUI()
#
#     def initUI(self):
#         self.setWindowTitle('Progress Dialog Example')
#         self.setGeometry(100, 100, 400, 100)
#
#         layout = QVBoxLayout()
#
#         self.progress_button = QPushButton('Start Progress')
#         self.progress_button.clicked.connect(self.show_progress_dialog)
#
#         layout.addWidget(self.progress_button)
#
#         self.setLayout(layout)
#
#     def show_progress_dialog(self):
#         self.progress_dialog = QDialog(self)
#         self.progress_dialog.setWindowTitle('Progress Dialog')
#         self.progress_dialog.setWindowModality(Qt.WindowModal)
#         self.progress_dialog.setGeometry(200, 200, 200, 200)
#
#         self.progress_bar = QProgressBar(self.progress_dialog)
#         self.progress_bar.setGeometry(30, 30, 140, 140)
#         self.progress_bar.setAlignment(Qt.AlignCenter)
#         self.progress_bar.setMinimum(0)
#         self.progress_bar.setMaximum(100)
#
#         self.worker_thread = WorkerThread()
#         self.worker_thread.update_progress.connect(self.update_progress)
#         self.worker_thread.start()
#
#         self.progress_dialog.exec_()
#
#     def update_progress(self, value):
#         self.progress_bar.setValue(value)
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     ex = ProgressDialogExample()
#     ex.show()
#     sys.exit(app.exec_())



# import sys
#
# from PyQt5.QtCore import Qt
# from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
# from qfluentwidgets import IndeterminateProgressRing, ProgressBar
# import sys
# from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
# from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QDialog, QProgressBar
#
# class Demo(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.vBoxLayout = QVBoxLayout(self)
#         self.inProgressBar = IndeterminateProgressRing(self)
#         self.vBoxLayout.addWidget(self.inProgressBar,0, Qt.AlignHCenter)
#         self.resize(400, 400)
# class ProgressDialogExample(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.initUI()
#     def initUI(self):
#         layout = QVBoxLayout()
#         self.progress_button = QPushButton('Start Progress')
#         self.progress_button.clicked.connect(self.show_progress_dialog)
#         layout.addWidget(self.progress_button)
#         self.setLayout(layout)
#     def show_progress_dialog(self):
#         self.demo = Demo()
#         self.demo.show()
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     ex = ProgressDialogExample()
#     ex.show()
#     sys.exit(app.exec_())

import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from qfluentwidgets import IndeterminateProgressRing, ProgressBar
import sys
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QDialog, QProgressBar
from utils.tools import Progress_inf

class Demo(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent=parent)
        self.vBoxLayout = QVBoxLayout(self)
        self.inProgressBar = IndeterminateProgressRing(self)
        self.vBoxLayout.addWidget(self.inProgressBar,0, Qt.AlignHCenter)
        self.resize(400, 400)
class ProgressDialogExample(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        layout = QVBoxLayout()
        self.progress_button = QPushButton('Start Progress')
        self.progress_button.clicked.connect(self.show_progress_dialog)
        layout.addWidget(self.progress_button)
        self.setLayout(layout)
    def show_progress_dialog(self):
        self.demo = Progress_inf()
        self.demo.show()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ProgressDialogExample()
    ex.show()
    sys.exit(app.exec_())