# -*- coding: utf-8 -*-
'''Updated 2021.02.17.'''

import pymssql
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

COMPLEX_LIST = ['냉열원 기동상태', '온열원 기동상태']


class HDC_DB:

    sql_names = ['SiteId', 'FacilityTypeId', 'FacilityCode',
            'PropertyId', 'CreatedDateTime', 'CurrentValue']

    def __init__(self, path='DB/', db=1):

        self.path = path  #? 관제점 정리 엑셀 파일 경로
        self.db_name = 'iBems_SU' if db == 1 else 'iBems_SU2'  #? 사용할 데이터베이스

        self.ts = timedelta(minutes=15)  #? 1 타임스텝

        self.load_id_db()


    def load_id_db(self):
        '''엑셀 파일로부터 관제점 정보 불러옴'''

        id_db = pd.read_excel(self.path+'대구은행관제점정리_2020.xlsx',
                                   header = 1,
                                   usecols = 'B:I')
        id_db = id_db[id_db.columns[[0, 1, 2, 3, 5, 6]]]
        id_db.rename(columns = {'Name': 'Name 1', 'Name.1': 'Name 2'},
                     inplace = True)

        self.id_db = id_db
        self.names_id = {
            (name1.strip(), name2.strip()): {
                'DBID': dbid,
                'FacilityTypeId': ftid,
                'FacilityCode': fc,
                'PropertyId': pid,
            }
            for dbid, ftid, fc, pid, name1, name2 \
            in id_db.values.tolist()
        }


    @staticmethod
    def time_to_string(year, month, day, hour=0, minute=0, for_table=False):
        if for_table:
            return f'{year}-{month:02}-{day:02} {hour:02}:{minute:02}:00'
        else:
            return f'{year}-{month:02}-{day:02} {hour:02}:{minute:02}:00.000'

    @staticmethod
    def string_to_time(string):
        time_tuple = (
            int(string[0:4]), int(string[5:7]), int(string[8:10]),
            int(string[11:13]), int(string[14:16])#, int(string[17:19])
        )
        return time_tuple

    @staticmethod
    def datetime_to_tuple(dt):
        return (dt.year, dt.month, dt.day, dt.hour, dt.minute)


    @staticmethod
    def time_compare(time1, time2, include=False):
        '''get_db_with_condition에서 사용할 시간 비교 함수'''
        if time1[0] == time2[0] and time1[1] == time2[1] and include:
            return True
        else:
            if time1[0] < time2[0]:
                return True
            elif time1[0] > time2[0]:
                return False
            elif time1[1] < time2[1]:
                return True
            else:
                return False


    def time_idx_bound(self, time, info, for_table=False):
        '''원하는 기간의 양끝 기간 string index 반환'''

        #? DataFrame인 경우 마지막 행을 무시하는 반면, (df.loc[time_1: time_2])
        #? SQL 쿼리 생성시 포함하기 때문 ('BETWEEN time_1 AND time_2')
        # offset = timedelta(0) if for_table else self.ts
        offset = self.ts

        if isinstance(info, int):
            #? info가 타임스텝 수를 의미하는 경우
            delta = self.ts * abs(info)

            if info > 0:
                time_1 = time
                time_2 = self.datetime_to_tuple(datetime(*time)+delta-offset)
            elif info < 0:
                time_2 = self.datetime_to_tuple(datetime(*time)-offset)
                time_1 = self.datetime_to_tuple(datetime(*time_2)-delta+offset)

        if isinstance(info, tuple):
            #? info가 두번째 시간 정보를 의미하는 경우
            if datetime(*time) > datetime(*info):
                temp = time
                time = info
                info = temp
            time_1 = time
            time_2 = self.datetime_to_tuple(datetime(*info)-offset)

        time_string_1 = self.time_to_string(*time_1, for_table=for_table)
        time_string_2 = self.time_to_string(*time_2, for_table=for_table)

        return time_string_1, time_string_2


    def generate_query(self, columns, dt_string):
        '''입력 정보를 바탕으로 쿼리문 생성'''

        sql_query = '\n'.join([  #? 아래의 문자열 리스트를 연결
            'SELECT *',
            f'FROM [{self.db_name}].[dbo].[BemsMonitoringPointHistory15min]',
            ' '.join([  #? 관제점 관련 조건문들 연결
                'WHERE',
                '(' + ' OR '.join([
                    '('+' AND '.join([
                        f'{k}={v}' for k, v in self.names_id[column].items()
                        if k != 'DBID'
                        ])+')' for column in columns
                ]) + ')',
                'AND',
                'siteid = 1',
                'AND',
                '(CreatedDateTime BETWEEN',  #? 기간 조건문
                '\''+dt_string[0]+'\'',
                'AND',
                '\''+dt_string[1]+'\')',
            ]),
        ])
        return sql_query


    def connect(self):
        '''SQL 서버에 연결'''
        self.conn = pymssql.connect(host='61.33.215.50', user='DGB_SU', \
                password='dgbsu123', database=self.db_name, charset='utf8')


    def close(self):
        '''서버와의 연결 종료'''
        self.conn.close()


    def retrieve_from_db(self, columns, time, info, nan='fill'):
        '''관제점 tuple 목록과 기간을 받아 DataFrame 반환'''
        # self.connect() 후에 사용

        new_columns = []
        for column in columns:
            if column[1] not in COMPLEX_LIST:
                if column not in new_columns:
                    new_columns.append(column)
            else:
                #? 복합 관제점인 경우 (ex. 냉열원 기동상태, 온열원 기동상태)
                if column[1] == '냉열원 기동상태':
                    for v in ['제빙운전 상태', '축단운전 상태', '병렬운전 상태', '냉단운전 상태']:
                        new_column = ('축열조_IST_101', v)
                        if new_column not in new_columns:
                            new_columns.append(new_column)
                if column[1] == '온열원 기동상태':
                    for k in ['온수보일러_B-101-1', '온수보일러_B-101-2']:
                        new_column = (k, '온수 보일러 운전 상태')
                        if new_column not in new_columns:
                            new_columns.append(new_column)
                if column[0][:3] == '공조기':
                    new_column = (column[0], '공조기 기동상태')
                    if new_column not in new_columns:
                        new_columns.append(new_column)
        columns = new_columns

        cursor = self.conn.cursor()

        dt_string = self.time_idx_bound(time, info)
        sql_query = self.generate_query(columns, dt_string)  #? 쿼리문 생성
        cursor.execute(sql_query)  #? 쿼리 요청

        row = cursor.fetchall()  #? 서버에서 받아온 데이터

        output_raw = pd.DataFrame(row, columns=self.sql_names)  #? DataFrame으로 만듦

        id_db = self.id_db[
            self.id_db['DBID'].isin(
                [str(self.names_id[names]['DBID']) for names in columns]
            )
        ]

        output_dbid_added = pd.merge(id_db, output_raw)

        output = pd.concat([
            output_dbid_added[
                output_dbid_added['DBID'] == self.names_id[names]['DBID']
            ]
            [['CreatedDateTime', 'CurrentValue']]
            .rename(columns={
                'CreatedDateTime': 'Time',
                'CurrentValue': self.names_id[names]['DBID']
            })
            .set_index('Time') for names in columns
        ], axis=1)

        #? 시간 축 생성 (output 행 전체가 결측인 경우 누락된 타임스텝이 발생할 수 있음)
        ts = pd.Series(0, index=pd.date_range(
                start=datetime(*self.string_to_time(dt_string[0])),
                end=datetime(*self.string_to_time(dt_string[1])), freq='15T'))

        output = pd.concat([output, ts], axis = 1).iloc[:, :-1]

        output.index = output.index.map(str)  #? index를 datetime에서 str으로

        #? 결측값 처리
        if nan == None:
            pass
        elif nan == 'fill':
            output.fillna(method='ffill', inplace=True)
            output.fillna(method='bfill', inplace=True)
            output.fillna(value=0, inplace=True)
        elif nan == 'drop':
            output.dropna(inplace=True)
        else:
            output.fillna(value=nan, inplace=True)

        return output


    def get_sequence(self, names, time, info, table=None):
        '''하나의 관제점 tuple과 기간을 받아 array 반환'''

        if names[1] not in COMPLEX_LIST:
            if isinstance(table, pd.DataFrame):
                time_string_1, time_string_2 = self.time_idx_bound(time, info, for_table=True)
                return np.array(table.loc[time_string_1:time_string_2, self.names_id[names]['DBID']])
            else:
                return np.array(self.retrieve_from_db([names], time, info)).flatten()
        else:
            #? 복합 관제점인 경우 (ex. 냉열원 기동상태, 온열원 기동상태)
            # output = self.retrieve_from_db([(names[0], _) for _ in COMPLEX[names[1]]], time, info)
            if names[0][:3] == '공조기':
                is_ahu = True
                ahu_op = self.get_sequence((names[0], '공조기 기동상태'), time, info, table)
            else:
                is_ahu = False

            if names[1] == '냉열원 기동상태':
                
                cool = [self.get_sequence(('축열조_IST_101', _), time, info, table) \
                        for _ in ['제빙운전 상태', '축단운전 상태', '병렬운전 상태', '냉단운전 상태']]
                cool_op = np.array([1 if (cool[1][_]>0 or cool[2][_]>0 or cool[3][_]>0) \
                        else 0 for _ in range(cool[0].shape[0])])
                '''
                for cool1, cool2, cool3 in zip(cool[1], cool[2], cool[3]):
                    if cool1 > 0 or cool2 > 0 or cool3 > 0: cool_op.append
                    if cool2 > 0: cool_op.append(2)
                    elif cool1 > 0: cool_op.append(1)
                    elif cool3 > 0: cool_op.append(3)
                    else: cool_op.append(0)
                '''
                if is_ahu:
                    return cool_op * ahu_op
                else:
                    return cool_op

            elif names[1] == '온열원 기동상태':

                heat = [self.get_sequence((f'온수보일러_B-101-{_}', '온수 보일러 운전 상태'), \
                        time, info, table) for _ in [1, 2]]
                # return heat[0]+heat[1]
                heat_op = np.array([1 if (heat[0][_]>0 or heat[1][_]>0) else 0 \
                    for _ in range(heat[0].shape[0])])
                '''
                for heat1, heat2 in zip(*heat):
                    if heat1+heat2 > 0: heat_op.append(1)
                    else: heat_op.append(0)
                '''
                if is_ahu:
                    return heat_op * ahu_op
                else:
                    return heat_op

        # return np.array(self.retrieve_from_db(columns, time, info))


    def get_db_with_condition(self, time, info, conditions, columns):
        '''원하는 기간동안, 특정 조건을 만족하는 데이터들만 반환'''

        # time_idx_1, time_idx_2 = self.time_idx_range(time, info)
        db_ids = [self.names_id[names]['DBID'] for names in columns]
        sub_db = self.retrieve_from_db(columns, time, info)

        masks = pd.Series(True, index=sub_db.index)  #? Boolean mask 생성

        for condition in conditions:

            names, op, num = condition

            if isinstance(names, tuple):
                #? ex. (('백엽상', '외기건구온도'), '<', 27)

                db_id = self.names_id[names]['DBID']

                if op == '==':
                    mask = (sub_db[db_id] == num)
                elif op == '!=':
                    mask = (sub_db[db_id] != num)
                elif op == '>':
                    mask = (sub_db[db_id] > num)
                elif op == '<':
                    mask = (sub_db[db_id] < num)
                elif op == '>=':
                    mask = (sub_db[db_id] >= num)
                elif op == '<=':
                    mask = (sub_db[db_id] <= num)
                elif op == 'in':
                    mask = (sub_db[db_id] >= num[0]) & (sub_db[db_id] <= num[1])
                elif op == 'not in':
                    mask = (sub_db[db_id] < num[0]) & (sub_db[db_id] > num[1])

            elif isinstance(names, str):

                if names.lower() == 'time':
                    #? ex. ('time', 'in', ((5, 15), [7, 0]))

                    time1, time2 = num
                    #? 경계를 포함할지 여부 (list면 포함, tuple이면 제외)
                    inc1, inc2 = isinstance(time1, list), isinstance(time2, list)

                    h_m_list = [self.string_to_time(idx)[3:] for idx in sub_db.index]

                    if op == 'in' or op == 'not in':

                        mask = [
                            self.time_compare(time1, h_m, inc1) \
                                & self.time_compare(h_m, time2, inc2)
                            for h_m in h_m_list
                        ]

                        if op == 'not in':
                            mask = [not m for m in mask]

            masks = masks & mask

        return sub_db.loc[masks, db_ids]

