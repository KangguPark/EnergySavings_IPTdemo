# -*- coding: utf-8 -*-
'''Updated 2021.02.04.'''

import utils.Time_Index as ti

import numpy as np
import pandas as pd
import pickle
import os

from datetime import time, datetime, timedelta

COMPLEX_LIST = ['온열원 기동상태']


class HDC_XL:

	@staticmethod
	def time_to_string(year, month, day, hour=0, minute=0):
		return f'{year}-{month:02}-{day:02} {hour:02}:{minute:02}:00'


	@staticmethod
	def string_to_time(string):
		time_tuple = (
			int(string[0:4]), int(string[5:7]), int(string[8:10]),
			int(string[11:13]), int(string[14:16])#, int(string[17:19])
		)
		return time_tuple


	@staticmethod
	def time_compare(time1, time2, include=False):
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


class HDC_IPARK_XL(HDC_XL):


	# TIMERANGE = [time(0, 0, 0) + timedelta(minutes=5)*hr for hr in range(60)]

	# DATERANGE = [datetime(2020, 10, 9)]
	DATERANGE = [datetime(2020, 10, 8) + timedelta(days=1)*dy \
		for dy in range((datetime(2020, 12, 10)-datetime(2020, 10, 8)).days+1)]
	# DATERANGE = [datetime(2020, 10, 8) + timedelta(days=1)*dy \
	#     for dy in range((datetime(2020, 10, 19)-datetime(2020, 10, 8)).days+1)] \
	#     + [datetime(2020, 10, 21) + timedelta(days=1)*dy \
	#     for dy in range((datetime(2020, 12, 10)-datetime(2020, 10, 21)).days+1)]

	def __init__(self, path='DB/', subpath='아이파크데이터/', refresh=False):
		#os.chdir(r'D:\IPTower\AHUOptimulControl\\')  # move directory to current file

		self.path = path
		self.subpath = subpath
		
		self.load_id_db(refresh)
		self.load_bems_db(refresh)


	@staticmethod
	def read_txt(filename):
		db = pd.read_csv(filename, delimiter = '\t', encoding = 'cp949')
		db.rename(columns = {'시각' : 'Time'}, inplace = True)
		db = db.set_index('Time')

		# ts = pd.Series(0, index=pd.date_range(
		# start=datetime(*HDC_XL.string_to_time(dt_string[0])),
		# end=datetime(*HDC_XL.string_to_time(dt_string[1])), freq="15T"))

		return db
   

	def get_id(self, name1, name2):
		'''기기 이름과 데이터 이름 받아서 id 반환하는 함수'''
		return self.id_db[(name1, name2)]


	def get_time_idx(self, year, month, day, hour=0, minute=0):
		label = self.time_to_string(year, month, day, hour, minute)
		return self.db.index.get_loc(label)


	def time_idx_range(self, time, info):
		'''원하는 기간의 시간 index를 반환하는 함수'''

		if isinstance(time, tuple):

			if isinstance(info, int):
				if info  > 0:
					time_idx_1 = self.get_time_idx(*time)
					time_idx_2 = time_idx_1 + info
				elif info < 0:
					time_idx_2 = self.get_time_idx(*time)
					time_idx_1 = time_idx_2 + info
				else:
					return []

			if isinstance(info, tuple):
				time_idx_1 = self.get_time_idx(*time)
				time_idx_2 = self.get_time_idx(*info)
				if time_idx_1 > time_idx_2:
					temp = time_idx_1
					time_idx_1 = time_idx_2
					time_idx_2 = temp

		elif isinstance(time, int):

			if isinstance(info, int):
				if info > 0:
					time_idx_1 = time
					time_idx_2 = time_idx_1 + info
				elif info < 0:
					time_idx_2 = time
					time_idx_1 = time_idx_2 + info
				else:
					return []

		return time_idx_1, time_idx_2


	def load_id_db(self, refresh):

		if refresh:

			id_db = pd.read_excel(self.path+'아이파크타워관제점정리_20201020.xlsx',
								  header = 1,
								  usecols = 'B:H')
			id_db = id_db[id_db.columns[[1, 2, 3]]]
			id_db.rename(columns = {'관제설비': 'Name 1', '관제점 이름': 'Name 2', '태그 명': 'Tag'},
						inplace = True)
			id_db.fillna(method='ffill', inplace=True)

			# self.id_db = id_db
			self.id_db = {(name1.strip(), name2.strip()): str(tag) for name1, name2, tag \
				in id_db.values.tolist()}

			with open(self.path+'ipark_ids.pkl', 'wb') as f:
				pickle.dump(self.id_db, f)

		else:
			with open(self.path+'ipark_ids.pkl', 'rb') as f:
				self.id_db = pickle.load(f)


	def load_bems_db(self, refresh):

		if refresh:

			txt_name_list = list(f'2020{dt.month:02}{dt.day:02}.txt' for dt in self.DATERANGE)
			
			db_list = [
				self.read_txt(self.path+self.subpath+filename) for filename in txt_name_list
			]

			db_raw = pd.concat(db_list, axis=0)
			ts_list = [datetime.strptime(_, '%Y/%m/%d %H:%M:%S') for _ in db_raw.index]
			db_raw.index = map(str, ts_list)

			db_new = pd.DataFrame(0,
				index   = pd.date_range(
					start = datetime(2020, 10, 9),
					end   = datetime(2020, 12, 10, 23, 45), freq='15T'),
				columns = db_raw.columns
			)
			db_new.index = db_new.index.map(str)

			for idx in db_new.index:
				current_dt = datetime(*ti.string_to_time(idx))
				closest_dt = min(ts_list, key=lambda x:abs(x-current_dt).total_seconds())
				db_new.loc[idx,:] = db_raw.loc[str(closest_dt),:].values
			
			# print(db_raw)
			# print(db_new)
			# db = db.head(-1)

			db = db_new
	
			db.to_pickle(self.path+'ipark_db.pkl')

		else:
			db = pd.read_pickle(self.path+'ipark_db.pkl')


		# db.fillna(method='ffill', inplace=True)
		# db.fillna(method='bfill', inplace=True)
		# db.fillna(value=-100, inplace=True)

		null_bound = [
			ti.time_to_string(2020, 10, 20, 11, 15),
			ti.time_to_string(2020, 10, 20, 23, 45)
		]
		db.loc[null_bound[0]:null_bound[1], :] = None

		self.db = db


	def retrieve_from_db(self, columns, time, info):
		time_idx_1, time_idx_2 = self.time_idx_range(time, info)
		
		db_ids = []
		for column in columns:
			if column[1] == '공조기 기동상태':
				db_ids.append(self.get_id(column[0], '급기팬 운전상태'))
			elif column[1] not in COMPLEX_LIST:
				db_ids.append(self.get_id(*column))
			else:
				if column[1] == '온열원 기동상태':
					for v in ['냉온수기1-1 운전상태', '냉온수기1-2 운전상태']:
						db_ids.append(self.get_id('냉온수기1', v))
			# db_ids = [self.get_id(*names) for names in columns]

		sub_db = self.db.iloc[time_idx_1: time_idx_2][db_ids]

		return sub_db


	def get_sequence(self, names, time, info, table=None):
		'''원하는 기간동안의 특정 데이터를 반환하는 함수'''
		#? table 변수는 HDC_SQL과의 호환성을 위해

		if names[1] not in COMPLEX_LIST:
			return np.array(self.retrieve_from_db([names], time, info)).flatten()
		else:
			if names[0][:3] == 'AHU':
				is_ahu = True
				ahu_op = self.get_sequence((names[0], '공조기 기동상태'), time, info)
			else:
				is_ahu = False
			
			if names[1] == '온열원 기동상태':
				heat = [self.get_sequence(('냉온수기1', f'냉온수기1-{_} 운전상태'), \
						time, info) for _ in [1, 2]]
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


if __name__ == '__main__':

	ipark_xl = HDC_IPARK_XL(refresh=False)
	
	sub_db_3 = ipark_xl.retrieve_from_db(
		[
			('냉온수기1', '냉온수기1-1 운전상태'),
			('냉온수기1', '냉온수기1-2 운전상태'),
			('냉온수기1', '냉온수기1-1 출구온도'),
			('냉온수기1', '냉온수기1-2 출구온도'),
			('냉온수기1', '냉온수기 공급펌프1 운전상태'),
			('냉온수기1', '냉온수기 공급펌프2 운전상태'),
			('냉온수기1', '냉온수기 공급펌프3 운전상태'),
			('냉온수기2', '냉온수기 운전상태'),
			('냉온수기2', '냉온수기 출구온도'),
		],
		# (2020, 11, 20, 10, 0), (2020, 12, 9, 18),
		(2020, 10, 9), (2020, 12, 10),
	)
	print()
	# print(sub_db_3.to_string())
	
	sub_db_4 = ipark_xl.get_sequence(
		('냉온수기1', '냉온수기1-1 운전상태'),
		(2020, 11, 20, 12, 0), 5
	)
	print()
	print(sub_db_4)

	sub_db_5 = ipark_xl.retrieve_from_db(
		[
			# ('AHU6 (지상3층)', 'AHU 급기(SA)온도'),
			# ('AHU6 (지상3층)', 'AHU 환기(RA)온도'),
			# ('AHU6 (지상3층)', 'AHU 외기(OA)온도'),
			('AHU6 (지상3층)', 'AHU 운전모드'),
			('AHU6 (지상3층)', '급기팬 운전상태'),
			('AHU6 (지상3층)', '환기팬 운전상태'),
		],
		(2020, 10, 9, 8, 0), (2020, 12, 10),
	)
	print()
	# print(sub_db_5.loc[sub_db_5.iloc[:,1]!=sub_db_5.iloc[:,2]])
	
	sub_db_6 = ipark_xl.retrieve_from_db(
		[
			('외부', '외기온도'),
			('외부', '외기습도'),
		],
		# (2020, 11, 20, 10, 0), (2020, 12, 9, 18),
		(2020, 10, 21), (2020, 10, 23),
	)
	print()
	print(sub_db_6.to_string())

	'''
	import matplotlib.pyplot as plt

	fig, axs = plt.subplots(3, 1, figsize=(12, 4), sharex=True, gridspec_kw={'height_ratios':[3,1,1]})
	axs = axs.ravel()
	axs[0].plot(sub_db_3.iloc[:, 2:4])
	axs[0].plot(sub_db_3.iloc[:, -1])
	axs[1].plot(sub_db_3.iloc[:, 0:2])
	axs[2].plot(sub_db_3.iloc[:, 4:7])
	axs[0].set_xticks([])
	plt.tight_layout()
	plt.show()
	'''




