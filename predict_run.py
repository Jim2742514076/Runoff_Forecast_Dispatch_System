# -*-coding = utf-8 -*-
# @Time : 2023/7/18 9:48
# @Author : 万锦
# @File : predict_run.py
# @Softwore : PyCharm

from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tushare as ts
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from qfluentwidgets import Dialog,IndeterminateProgressRing,IndeterminateProgressBar
from torch.utils.data import TensorDataset
from tqdm import tqdm
from utils.tools import Progress_inf
from torchsummary import summary
from ui.predict import Ui_MainWindow

ui,_ = loadUiType("./ui/predict.ui")

#初始化配置
class Config():
    # data_path = 'test.csv'
    timestep = 1  # 时间步长，就是利用多少时间窗口
    batch_size = 32  # 批次大小
    feature_size = 5  # 每个步长对应的特征数量
    hidden_size = 256  # 隐层大小
    # output_size = 2  # 由于是多输出任务，最终输出层大小为2
    output_size = 1
    num_layers = 2  # lstm的层数
    epochs = 10 # 迭代轮数
    best_loss = 0 # 记录损失
    learning_rate = 0.0003 # 学习率
    model_name = 'lstm' # 模型名称
    save_path = './{}.pth'.format(model_name) # 最优模型保存路径


# 定义LSTM网络
class LSTM(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        # feature_size为特征维度，就是每个时间点对应的特征数量，这里为1
        # self.feature_size = feature_size
        self.hidden_size = hidden_size  # 隐层大小
        self.num_layers = num_layers  # lstm层数

        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]  # 获取批次大小

        # 初始化隐层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
            c_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0, c_0 = hidden

        # LSTM运算
        output, (h_0, c_0) = self.lstm(x, (h_0, c_0))

        # 全连接层
        output = self.fc(output)  # 形状为batch_size * timestep, 1

        # 我们只需要返回最后一个时间片的数据即可
        return output[:, -1, :]



class Form_predict(QMainWindow,ui):
    def __init__(self):
        super(Form_predict, self).__init__()
        self.setupUi(self)
        self.setObjectName("predict_form")
        #数据源
        self.data = 0
        self.data_deal = 0
        #模型
        self.model = 0

        #训练集、测试集输出值
        self.y_true_train = 0
        self.y_true_test = 0
        self.y_pre_train = 0
        self.y_pre_test = 0

        self.initialize_lineedit()
        self.initialize_combox()

        self.handle_buttom()


    # 初始化下拉框
    def initialize_combox(self):
        # 从列表中添加下拉选项
        self.ComboBox.addItems(["LSTM","BP","GRU"])
        # 设置显示项目
        self.ComboBox.setCurrentIndex(0)

    # 设置输入控件
    def initialize_lineedit(self):
        # 设置输入值只能为整数
        validator_int = QIntValidator()
        # 设置范围为0.0到1.0，小数点后最多4位
        validator_float = QDoubleValidator()
        validator_float.setRange(0.0, 1.0, decimals=4)

        # 预报因子数目
        self.LineEdit.setValidator(validator_int)
        self.LineEdit.setPlaceholderText("5")
        # 神经元个数
        self.LineEdit_7.setValidator(validator_int)
        self.LineEdit_7.setPlaceholderText("4")
        # 隐藏层大小
        self.LineEdit_8.setValidator(validator_int)
        self.LineEdit_8.setPlaceholderText("256")
        # 时间步长
        self.LineEdit_9.setValidator(validator_int)
        self.LineEdit_9.setPlaceholderText("1")
        # 迭代次数
        self.LineEdit_10.setValidator(validator_int)
        self.LineEdit_10.setPlaceholderText("10")
        # 批次大小
        self.LineEdit_11.setValidator(validator_int)
        self.LineEdit_11.setPlaceholderText("32")
        # 训练集比例
        self.LineEdit_2.setValidator(validator_float)
        self.LineEdit_2.setPlaceholderText("0.7")
        # 学习率
        self.LineEdit_17.setValidator(validator_float)
        self.LineEdit_17.setPlaceholderText("0.0003")

    # 功能控制
    def handle_buttom(self):
        self.PushButton_5.clicked.connect(self.add_data)
        self.PushButton_2.clicked.connect(self.deal_data)
        self.PushButton.clicked.connect(self.model_train)
        self.PushButton_4.clicked.connect(self.draw_img)
        self.PushButton_3.clicked.connect(self.model_valid)
        self.PushButton_6.clicked.connect(self.output_data)

    # 输出检验值
    def model_valid(self):
        def mape(y_true, y_pred):
            return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

        # 定义函数计算Nash-Sutcliffe效率系数
        def nash_sutcliffe(obs, sim):
            return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

        # 计算均方误差
        mse = mean_squared_error(self.y_true_test, self.y_pre_test)
        mae = mean_absolute_error(self.y_true_test, self.y_pre_test)
        nse = nash_sutcliffe(self.y_true_test, self.y_pre_test)
        r2 = r2_score(self.y_true_test, self.y_pre_test)
        self.LineEdit_3.setText(str(mse))
        self.LineEdit_4.setText(str(mae))
        self.LineEdit_5.setText(str(nse))
        self.LineEdit_6.setText(str(r2))

    def test(self):
        self.demo = Progress_inf()
        self.demo.show()
        self.BodyLabel.setText("test")



    #导入数据
    def add_data(self):
        fname, _ = QFileDialog.getOpenFileName(self, "打开文件", '.', '数据文件(*.xlsx)')
        if fname:
            # self.statusBar().showMessage("数据加载中......")
            df = pd.read_excel(fname, index_col=0)
            self.data = df
            # self.year_index = df.index.values
            data = df.values
            # 表格加载数据
            # 设置行列，设置表头
            tmp = [str(_) for _ in df.columns.tolist()]
            tmp2 = [str(_) for _ in df.index.tolist()]
            self.TableWidget.setRowCount(len(data))
            self.TableWidget.setColumnCount(len(data[0]))
            self.TableWidget.setHorizontalHeaderLabels(tmp)
            self.TableWidget.setVerticalHeaderLabels(tmp2)
            # 设置单元格对齐方式为中心
            cell_alignment = Qt.AlignCenter
            # 表格加载内容
            for row, form in enumerate(data):
                for column, item in enumerate(form):
                    use_item = QTableWidgetItem(str(round(item, 2)))
                    use_item.setTextAlignment(cell_alignment)
                    self.TableWidget.setItem(row, column, use_item)

    # 显示对话框信息
    def showDialog(self, name=""):
        title = '操作成功'
        content = f"""{name}数据已经成功加载，点击OK进行下一步"""
        w = Dialog(title, content, self)
        w.setTitleBarVisible(False)
        if w.exec():
            pass
        else:
            pass
    # 操作成功
    def show_success(self,name=""):
        title = '操作成功'
        content = f"""{name}"""
        w = Dialog(title, content, self)
        w.setTitleBarVisible(False)
        if w.exec():
            pass
        else:
            pass
    # 警示对话框信息
    def warning_dialog(self, name=""):
        title = '操作失败'
        content = f"""{name}"""
        w = Dialog(title, content, self)
        w.setTitleBarVisible(False)
        if w.exec():
            pass
        else:
            pass

    #数据预处理
    def deal_data(self):
        if isinstance(self.data, pd.DataFrame):
            scaler_model = MinMaxScaler()
            self.data_deal = scaler_model.fit_transform(np.array(self.data))
            # print(type(self.data_deal))
            self.showDialog(name="数据预处理")
        else:
            self.warning_dialog(name="数据未导入")

    #模型训练(lstm)
    def model_train(self):
        self.statusBar().showMessage("模型训练中...")
        #实例化配置类
        config = Config()
        print(config.timestep,config.feature_size,config.output_size)
        def split_data(data, timestep, feature_size, output_size):
            dataX = []  # 保存X
            dataY = []  # 保存Y

            # 将整个窗口的数据保存到X中，将未来一天保存到Y中
            for index in range(len(data) - timestep - 1):
                dataX.append(data[index: index + timestep])
                dataY.append(data[index + timestep: index + timestep + output_size][:, -1].tolist())

            dataX = np.array(dataX)
            dataY = np.array(dataY)

            # 获取训练集大小
            train_size = int(np.round(0.8 * dataX.shape[0]))

            # 划分训练集、测试集
            x_train = dataX[: train_size, :].reshape(-1, timestep, feature_size)
            y_train = dataY[: train_size].reshape(-1, output_size)

            x_test = dataX[train_size:, :].reshape(-1, timestep, feature_size)
            y_test = dataY[train_size:].reshape(-1, output_size)

            return [x_train, y_train, x_test, y_test]

        # 3.获取训练数据   x_train: 170000,30,1   y_train:170000,7,1
        if isinstance(self.data_deal, np.ndarray):
            if self.LineEdit_9.text():
                config.timestep = int(self.LineEdit_9.text())
            if self.LineEdit.text():
                config.feature_size = int(self.LineEdit.text())
            config.output_size = 1
            print(config.timestep, config.feature_size, config.output_size)
            x_train, y_train, x_test, y_test = split_data(self.data_deal, config.timestep, config.feature_size, config.output_size)
            self.showDialog(name="训练集、测试集")
            # 4.将数据转为tensor
            x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
            y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
            x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
            y_test_tensor = torch.from_numpy(y_test).to(torch.float32)

            # 5.形成训练数据集
            train_data = TensorDataset(x_train_tensor, y_train_tensor)
            test_data = TensorDataset(x_test_tensor, y_test_tensor)
            # 6.将数据加载成迭代器
            train_loader = torch.utils.data.DataLoader(train_data,
                                                       config.batch_size,
                                                       False)

            test_loader = torch.utils.data.DataLoader(test_data,
                                                      config.batch_size,
                                                      False)
            # 7.实例化LSTM模型
            model = LSTM(config.feature_size, config.hidden_size, config.num_layers, config.output_size)  # 定义LSTM网络
            loss_function = nn.MSELoss()  # 定义损失函数
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)  # 定义优化器

            #弹窗进度条无法设置，采用信息提示栏显示
            # progressinf = Progress_inf()
            # progressinf.show()

            # 8.模型训练
            for epoch in range(config.epochs):
                model.train()
                running_loss = 0
                train_bar = tqdm(train_loader)  # 形成进度条
                for data in train_bar:
                    x_train, y_train = data  # 解包迭代器中的X和Y
                    optimizer.zero_grad()
                    y_train_pred = model(x_train)
                    loss = loss_function(y_train_pred, y_train.reshape(-1, 1))
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                             config.epochs,
                                                                             loss)
                # 模型验证
                model.eval()
                test_loss = 0
                with torch.no_grad():
                    test_bar = tqdm(test_loader)
                    for data in test_bar:
                        x_test, y_test = data
                        y_test_pred = model(x_test)
                        test_loss = loss_function(y_test_pred, y_test.reshape(-1, 1))

                if test_loss < config.best_loss:
                    config.best_loss = test_loss
                    torch.save(model.state_dict(), config.save_path)
            # print('Finished Training')
            # self.progressinf.close()
            self.statusBar().showMessage(" ")
            # self.BodyLabel.setText(str(model))

            self.model = model
            scaler = MinMaxScaler()
            scaler.fit_transform(np.array(self.data.iloc[:,-1]).reshape(-1, 1))
            self.y_pre_train = scaler.inverse_transform((model(x_train_tensor).detach().numpy()[:, 0]).reshape(-1, 1))
            self.y_true_train = scaler.inverse_transform(y_train_tensor.detach().numpy().reshape(-1, 1)[:])
            self.y_pre_test = scaler.inverse_transform(model(x_test_tensor).detach().numpy()[: , 0].reshape(-1, 1))
            self.y_true_test = scaler.inverse_transform(y_test_tensor.detach().numpy().reshape(-1, 1)[:])
            self.show_success(name="模型训练结束")
        else:
            self.warning_dialog(name="数据未进行预处理")



    # 输出检验值
    def model_valid(self):
        def mape(y_true, y_pred):
            return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

        # 定义函数计算Nash-Sutcliffe效率系数
        def nash_sutcliffe(obs, sim):
            return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

        # 计算均方误差
        mse = mean_squared_error(self.y_true_test, self.y_pre_test)
        mae = mean_absolute_error(self.y_true_test, self.y_pre_test)
        nse = nash_sutcliffe(self.y_true_test, self.y_pre_test)
        r2 = r2_score(self.y_true_test, self.y_pre_test)
        self.LineEdit_3.setText(str(mse))
        self.LineEdit_4.setText(str(mae))
        self.LineEdit_5.setText(str(nse))
        self.LineEdit_6.setText(str(r2))

    #绘制图像
    def draw_img(self):
        plt.figure(figsize=(12, 8))
        plt.plot(self.y_pre_train,"b",label="pre")
        plt.plot(self.y_true_train,"r",label="true")
        plt.legend()
        # 获取当前图表对象
        fig = plt.gcf()
        # 设置图片尺寸为 7x5 英寸
        fig.set_size_inches(6, 4)
        fig.savefig("./imgs/train.png")
        time.sleep(0.5)
        # 设置图片
        self.PixmapLabel_3.setPixmap(QPixmap("./imgs/train.png"))
        self.PixmapLabel_3.setScaledContents(True)  # 图片自适应窗口大小
        self.PixmapLabel_3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.PixmapLabel_3.setAlignment(Qt.AlignCenter)  # 图片居中显示

        plt.figure(figsize=(12, 8))
        plt.plot(self.y_pre_test, "b", label="pre")
        plt.plot(self.y_true_test, "r", label="true")
        plt.legend()
        # 获取当前图表对象
        fig = plt.gcf()
        # 设置图片尺寸为 7x5 英寸
        fig.set_size_inches(6, 4)
        fig.savefig("./imgs/test.png")
        time.sleep(0.5)
        # 设置图片
        self.PixmapLabel_4.setPixmap(QPixmap("./imgs/test.png"))
        self.PixmapLabel_4.setScaledContents(True)  # 图片自适应窗口大小
        self.PixmapLabel_4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.PixmapLabel_4.setAlignment(Qt.AlignCenter)  # 图片居中显示

    # 导出数据
    def output_data(self):
        # print(type(self.y_pre_train))
        if self.y_pre_train is not 0:
            train_results = pd.DataFrame({"pre": self.y_pre_train.flatten(), "true": self.y_true_train.flatten()})
            test_results = pd.DataFrame({"pre": self.y_pre_test.flatten(), "true": self.y_true_test.flatten()})
            # 保存数据
            path = QFileDialog.getSaveFileName(self, "保存训练集文件", "./", ("结果(*.xlsx)"))
            if path[0]:
                train_results.to_excel(path[0])
                # 保存数据
                path = QFileDialog.getSaveFileName(self, "保存测试集文件", "./", ("结果(*.xlsx)"))
                if path[0]:
                    test_results.to_excel(path[0])
        else:
            self.warning_dialog("训练集、测试集数据未生成")


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    mainwindow = Form_predict()
    mainwindow.setWindowTitle("来水预测")
    mainwindow.setWindowIcon(QIcon("./icons/Predict_white.svg"))
    mainwindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()