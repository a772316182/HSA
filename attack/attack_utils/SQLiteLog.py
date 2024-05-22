import datetime
import sqlite3


class SqliteDBLog:
    def __init__(self, db_name='OpenHGNN_v6_target'):
        self.db_name = db_name

    def create_table(self, cursor, table, data_dict):
        # 获取字典的键和值类型，用于创建表格
        columns = ', '.join([f'{key} {type(value).__name__}' for key, value in data_dict.items()])
        cursor.execute(f'CREATE TABLE IF NOT EXISTS {table} ({columns})')

    def insert_new(self, collection_name, data):
        d1 = datetime.date.today()
        data['year'] = d1.year
        data['month'] = d1.month
        data['day'] = d1.day
        # 连接到SQLite数据库（如果不存在则会创建）
        conn = sqlite3.connect(self.db_name + '.db')

        # 创建一个游标对象
        cursor = conn.cursor()

        # 创建一个表格（如果不存在）
        self.create_table(cursor, collection_name, data)

        # 获取字典的键和值，用于构建动态的插入语句
        keys = ', '.join(data.keys())
        values = ', '.join(['?' for _ in range(len(data))])

        # 构建插入语句，并执行
        insert_query = f'INSERT INTO {collection_name} ({keys}) VALUES ({values})'
        cursor.execute(insert_query, tuple(data.values()))

        # 提交更改并关闭连接
        conn.commit()
        conn.close()
