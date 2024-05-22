import datetime
from urllib.parse import quote_plus

from pymongo import MongoClient


class MongoDBLog:
    def __init__(self, db_name='runoobdb'):
        self.db_name = db_name

    def insert_new(self, collection_name, data):
        user = 'haoran'
        password = '19971216Ll@'
        host = '39.107.52.16'
        port = '8080'

        uri = "mongodb://%s:%s@%s:%s" % (
            quote_plus(user), quote_plus(password), host, port)

        myclient = MongoClient(uri)
        mydb = myclient[self.db_name]
        mycollection = mydb[collection_name]
        d1 = datetime.date.today()
        data['year'] = d1.year
        data['month'] = d1.month
        data['day'] = d1.day
        result = mycollection.insert_one(data)
        return result


if __name__ == '__main__':
    user = 'haoran'
    password = '19971216Ll@'
    host = '117.72.37.153'
    port = '8080'
    uri = "mongodb://%s:%s@%s:%s" % (
        quote_plus(user), quote_plus(password), host, port)
    myclient = MongoClient(uri)
    mydb = myclient["runoobdb"]
    mycollection = mydb["sites"]
    stu1 = {'id': '001', 'name': 'zhangsan', 'age': 10}
    result = mycollection.insert_one(stu1)
