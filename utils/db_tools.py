# -*-coding = utf-8 -*-
# @Time : 2023/7/18 10:18
# @Author : 万锦
# @File : db_tools.py
# @Softwore : PyCharm

#构建数据库工具类
import pymysql
#获取连接
def get_conn():
    try:
        conn = pymysql.connect(host='localhost',
                               database='rainoff_predict_dispatch',
                               user="root",
                               password="123456")
        cursor = conn.cursor()
        # print("数据库连接成功")
        return conn,cursor
    except Exception:
        print("数据库连接异常")

def close_conn(conn,cursor):
    try:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    except Exception:
        print("数据库关闭异常")

if __name__ == '__main__':

    conn,_ = get_conn()

    close_conn(conn,conn.cursor())