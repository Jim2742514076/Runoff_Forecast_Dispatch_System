# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'predict.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1628, 1022)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.CardWidget = CardWidget(self.centralwidget)
        self.CardWidget.setMaximumSize(QtCore.QSize(250, 16777215))
        self.CardWidget.setObjectName("CardWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.CardWidget)
        self.gridLayout.setObjectName("gridLayout")
        self.LineEdit_2 = LineEdit(self.CardWidget)
        self.LineEdit_2.setObjectName("LineEdit_2")
        self.gridLayout.addWidget(self.LineEdit_2, 4, 1, 1, 1)
        self.LineEdit = LineEdit(self.CardWidget)
        self.LineEdit.setObjectName("LineEdit")
        self.gridLayout.addWidget(self.LineEdit, 2, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.CardWidget)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 4, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.CardWidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)
        self.PushButton_5 = PushButton(self.CardWidget)
        self.PushButton_5.setMinimumSize(QtCore.QSize(0, 50))
        self.PushButton_5.setObjectName("PushButton_5")
        self.gridLayout.addWidget(self.PushButton_5, 1, 0, 1, 2)
        self.ComboBox = ComboBox(self.CardWidget)
        self.ComboBox.setObjectName("ComboBox")
        self.gridLayout.addWidget(self.ComboBox, 3, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.CardWidget)
        self.label.setMinimumSize(QtCore.QSize(0, 50))
        self.label.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: rgb(85, 170, 255);")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 2)
        self.label_3 = QtWidgets.QLabel(self.CardWidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 3, 0, 1, 1)
        self.gridLayout_3.addWidget(self.CardWidget, 1, 0, 1, 1)
        self.CardWidget_2 = CardWidget(self.centralwidget)
        self.CardWidget_2.setMaximumSize(QtCore.QSize(16777215, 500))
        self.CardWidget_2.setObjectName("CardWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.CardWidget_2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.TableWidget = TableWidget(self.CardWidget_2)
        self.TableWidget.setObjectName("TableWidget")
        self.TableWidget.setColumnCount(0)
        self.TableWidget.setRowCount(0)
        self.verticalLayout_2.addWidget(self.TableWidget)
        self.gridLayout_3.addWidget(self.CardWidget_2, 1, 1, 1, 1)
        self.CardWidget_5 = CardWidget(self.centralwidget)
        self.CardWidget_5.setObjectName("CardWidget_5")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.CardWidget_5)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_10 = QtWidgets.QLabel(self.CardWidget_5)
        self.label_10.setMinimumSize(QtCore.QSize(0, 50))
        self.label_10.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.label_10.setFont(font)
        self.label_10.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: rgb(85, 170, 255);")
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.verticalLayout_3.addWidget(self.label_10)
        self.PushButton_2 = PushButton(self.CardWidget_5)
        self.PushButton_2.setMinimumSize(QtCore.QSize(0, 50))
        self.PushButton_2.setObjectName("PushButton_2")
        self.verticalLayout_3.addWidget(self.PushButton_2)
        self.PushButton = PushButton(self.CardWidget_5)
        self.PushButton.setMinimumSize(QtCore.QSize(0, 50))
        self.PushButton.setObjectName("PushButton")
        self.verticalLayout_3.addWidget(self.PushButton)
        self.PushButton_3 = PushButton(self.CardWidget_5)
        self.PushButton_3.setMinimumSize(QtCore.QSize(0, 50))
        self.PushButton_3.setObjectName("PushButton_3")
        self.verticalLayout_3.addWidget(self.PushButton_3)
        self.PushButton_4 = PushButton(self.CardWidget_5)
        self.PushButton_4.setMinimumSize(QtCore.QSize(0, 50))
        self.PushButton_4.setObjectName("PushButton_4")
        self.verticalLayout_3.addWidget(self.PushButton_4)
        self.PushButton_6 = PushButton(self.CardWidget_5)
        self.PushButton_6.setMinimumSize(QtCore.QSize(0, 50))
        self.PushButton_6.setObjectName("PushButton_6")
        self.verticalLayout_3.addWidget(self.PushButton_6)
        self.gridLayout_3.addWidget(self.CardWidget_5, 2, 0, 1, 1)
        self.CardWidget_3 = CardWidget(self.centralwidget)
        self.CardWidget_3.setMaximumSize(QtCore.QSize(250, 16777215))
        self.CardWidget_3.setObjectName("CardWidget_3")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.CardWidget_3)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_7 = QtWidgets.QLabel(self.CardWidget_3)
        self.label_7.setMinimumSize(QtCore.QSize(0, 50))
        self.label_7.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: rgb(85, 170, 255);")
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.gridLayout_2.addWidget(self.label_7, 0, 0, 1, 2)
        self.label_5 = QtWidgets.QLabel(self.CardWidget_3)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 1, 0, 1, 1)
        self.LineEdit_3 = LineEdit(self.CardWidget_3)
        self.LineEdit_3.setObjectName("LineEdit_3")
        self.gridLayout_2.addWidget(self.LineEdit_3, 1, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.CardWidget_3)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.gridLayout_2.addWidget(self.label_6, 2, 0, 1, 1)
        self.LineEdit_4 = LineEdit(self.CardWidget_3)
        self.LineEdit_4.setObjectName("LineEdit_4")
        self.gridLayout_2.addWidget(self.LineEdit_4, 2, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.CardWidget_3)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.gridLayout_2.addWidget(self.label_8, 3, 0, 1, 1)
        self.LineEdit_5 = LineEdit(self.CardWidget_3)
        self.LineEdit_5.setObjectName("LineEdit_5")
        self.gridLayout_2.addWidget(self.LineEdit_5, 3, 1, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.CardWidget_3)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.gridLayout_2.addWidget(self.label_9, 4, 0, 1, 1)
        self.LineEdit_6 = LineEdit(self.CardWidget_3)
        self.LineEdit_6.setObjectName("LineEdit_6")
        self.gridLayout_2.addWidget(self.LineEdit_6, 4, 1, 1, 1)
        self.gridLayout_3.addWidget(self.CardWidget_3, 3, 0, 1, 1)
        self.CaptionLabel = CaptionLabel(self.centralwidget)
        self.CaptionLabel.setMinimumSize(QtCore.QSize(0, 60))
        self.CaptionLabel.setMaximumSize(QtCore.QSize(16777215, 60))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(False)
        font.setWeight(50)
        self.CaptionLabel.setFont(font)
        self.CaptionLabel.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: rgb(85, 170, 255);")
        self.CaptionLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.CaptionLabel.setObjectName("CaptionLabel")
        self.gridLayout_3.addWidget(self.CaptionLabel, 0, 0, 1, 2)
        self.CardWidget_4 = CardWidget(self.centralwidget)
        self.CardWidget_4.setMaximumSize(QtCore.QSize(16777215, 1000))
        self.CardWidget_4.setObjectName("CardWidget_4")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.CardWidget_4)
        self.verticalLayout.setObjectName("verticalLayout")
        self.PixmapLabel = PixmapLabel(self.CardWidget_4)
        self.PixmapLabel.setObjectName("PixmapLabel")
        self.verticalLayout.addWidget(self.PixmapLabel)
        self.gridLayout_3.addWidget(self.CardWidget_4, 2, 1, 2, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1628, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_4.setText(_translate("MainWindow", "训练集比例："))
        self.label_2.setText(_translate("MainWindow", "预报因子："))
        self.PushButton_5.setText(_translate("MainWindow", "数据导入"))
        self.label.setText(_translate("MainWindow", "初始化面板"))
        self.label_3.setText(_translate("MainWindow", "预报模型："))
        self.label_10.setText(_translate("MainWindow", "控制面板"))
        self.PushButton_2.setText(_translate("MainWindow", "数据预处理"))
        self.PushButton.setText(_translate("MainWindow", "模型训练"))
        self.PushButton_3.setText(_translate("MainWindow", "模型检验"))
        self.PushButton_4.setText(_translate("MainWindow", "模型预报"))
        self.PushButton_6.setText(_translate("MainWindow", "结果导出"))
        self.label_7.setText(_translate("MainWindow", "模型检验"))
        self.label_5.setText(_translate("MainWindow", "MSE："))
        self.label_6.setText(_translate("MainWindow", "MAPE："))
        self.label_8.setText(_translate("MainWindow", "NSE："))
        self.label_9.setText(_translate("MainWindow", "R："))
        self.CaptionLabel.setText(_translate("MainWindow", "预报模块"))
from qfluentwidgets import CaptionLabel, CardWidget, ComboBox, LineEdit, PixmapLabel, PushButton, TableWidget
