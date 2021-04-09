import mysql.connector
import pandas as pd
from mysql.connector import errorcode
from sqlalchemy import create_engine, inspect

from model.util import convert_dtypes

TABLES = {
    'stock_idx': (
        "CREATE TABLE IF NOT EXISTS `stock_index` ("
        "  `stock_code` char(4) NOT NULL,"
        "  `start` datetime NOT NULL,"
        "  `end` datetime NOT NULL"
        ") ENGINE=InnoDB"
    )
}


class DBConnector:
    def __init__(self, config):
        self.config = config
        # todo create db and tables

    def conn(self):
        try:
            cnx = mysql.connector.connect(**self.config)
            cnx.autocommit = True
            return cnx
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                raise Exception("Wrong username or password", err)
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database does not exist, trying to create...")
                self.create_db()
                return self.conn()
            else:
                raise Exception(err)

    # noinspection SqlNoDataSourceInspection
    def create_db(self):
        cnx = self.conn()
        try:
            with cnx.cursor() as cursor:
                sql = "CREATE DATABASE IF NOT EXISTS {0};".format(self.config['database'])
                cursor.execute(sql)
        except mysql.connector.Error as err:
            if err.errno != errorcode.ER_DB_CREATE_EXISTS:
                raise Exception('Fail to create database', err)
        finally:
            cnx.close()

    def execute_sql(self, sql, params=()):
        cnx = self.conn()
        try:
            with cnx.cursor() as cursor:
                cursor.execute(sql, params)
        except Exception as e:
            print(e)
        finally:
            cnx.close()

    def engine(self):
        return create_engine(
            'mysql+mysqlconnector://{0}:{1}@{2}/{3}'.format(
                self.config['user'],
                self.config['password'],
                self.config['host'],
                self.config['database']
            )
        )

    def sql_within_df(self, data, tbl_name, primary_keys="", if_exists="append"):
        if data.shape[0] == 0:
            print("No data in this response. Return in case to create a table with all columns NVARCHAR")
            return
        data.to_sql(
            name=tbl_name,
            con=self.engine(),
            if_exists=if_exists,
            index=False,
            dtype=convert_dtypes(dict(data.dtypes))
        )
        check = inspect(self.engine())
        if check.get_primary_keys(tbl_name) == [] and len(primary_keys) > 0:
            add_pk_sql = 'ALTER TABLE {0} ADD PRIMARY KEY ({1})'.format(tbl_name, primary_keys)
            self.execute_sql(add_pk_sql)

    def read_within_df(self, sql):
        return pd.read_sql(sql=sql, con=self.engine())
