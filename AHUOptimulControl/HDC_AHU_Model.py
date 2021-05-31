# -*- coding: utf-8 -*-

import utils.Time_Index as ti
from utils.Plot_Result import plot_ctrl_result

import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import pickle
import re
import os.path

from datetime import datetime, timedelta

#physical_devices = tf.config.list_physical_devices('GPU') 
#tf.config.experimental.set_memory_growth(physical_devices[0], True)


class HDC_AHU_COMPONENT:

	TS = (1, 1)  # timestep (past, future)

	MAX_TARGET_TS = 1
	MIN_ON_TARGET_TS = 1

	input_past = []
	input_control = []
	output = []

	def __init__(self, cool, source, save_loc='Models/', name=None,
			bounds=(), now=datetime.now(), db=None, refresh=False):
		
		self.cool = cool
		if name:
			self.name = name
		else:
			self.name = 'ahu_' + source[-3:]
		self.source = source
		self.save_loc = save_loc
		self.bounds = bounds
		self.generated = ti.now(now)
		self.db = db
		self.info = {}

		self.input_past = [v if v[1][0] is not 'AHU' else (v[0], (source, v[1][1])) \
			for v in self.__class__.input_past]
		self.input_control = [v if v[0] is not 'AHU' else (source, v[1]) \
			for v in self.__class__.input_control]
		self.output = [v if v[0] is not 'AHU' else (source, v[1]) \
			for v in self.__class__.output]

		self.TS = self.__class__.TS

		self.input_size = sum([v[0] for v in self.input_past]) \
						  + self.TS[1]*len(self.input_control)
		self.output_size = self.TS[1]*len(self.output)

		if not os.path.isdir(self.save_loc): os.makedirs(self.save_loc)

		if os.path.isfile(self.save_loc+self.name+'.h5') and not refresh:
			self.load_model()
		else:
			self.model = None
			self.scaler_x = MinMaxScaler()
			self.scaler_y = MinMaxScaler()

			self.generate_model(self.input_size, self.output_size)
			self.initialize_model()


	def summary(self):
		'''View the summary of the module'''

		max_len = int(max(
			[np.floor(np.log10(s[0]))+1 for s in self.input_past]+
			[np.floor(np.log10(s))+1 for s in self.TS]
		))

		output_str = '\n'.join([
			'\n==== Model Summary ==========\n',
			' < Name > ' + self.name,
			'',
			' < Type > ' + self.__class__.__name__,
			'',
			' < Inputs - Past >',
			'\n'.join([f'  - [{v[0]:>{max_len}} ts] ({v[1][0]}, {v[1][1]})' \
				for v in self.input_past]),
			'',
			' < Inputs - Control >',
			'\n'.join([f'  - [{self.TS[1]:>{max_len}} ts] ({v[0]}, {v[1]})' \
				for v in self.input_control]),
			'',
			' < Outputs >',
			'\n'.join([f'  - [{self.TS[1]:>{max_len}} ts] ({v[0]}, {v[1]})' \
				for v in self.output]),
			'\n=============================\n'
		])
		
		print(output_str)


	def save_info(self, now):

		info = {
			'Timestamp': now,
			'TS':        self.TS,
			'I_P':       self.input_past,
			'I_C':       self.input_control,
			'O':         self.output,
		}

		self.info = info

		with open(self.save_loc+self.name+'_info.pkl', 'wb') as f:
			pickle.dump(info, f)


	#? ======== Model Management =======================

	def save_model(self):
		'''ANN 모델과 scaler들을 파일로 저장'''
		self.model.save(self.save_loc+self.name+'.h5')
		joblib.dump(self.scaler_x, self.save_loc+self.name+'_x.gz')
		joblib.dump(self.scaler_y, self.save_loc+self.name+'_y.gz')


	def load_model(self):
		'''ANN 모델과 scaler들을 파일에서 불러옴'''
		self.model = tf.keras.models.load_model(self.save_loc+self.name+'.h5')
		self.scaler_x = joblib.load(self.save_loc+self.name+'_x.gz')
		self.scaler_y = joblib.load(self.save_loc+self.name+'_y.gz')


	def update_model(self, now='now', alert=True):

		now_dt = ti.now(now)
		
		with open(self.save_loc+self.name+'_info.pkl', 'rb') as f:
			model_info = pickle.load(f)
		model_dt = model_info['Timestamp']

		update_queue, update_dt = ti.generate_update_queue(model_dt, now_dt, self.cool, self.bounds)

		if len(update_queue) > 0:

			if alert:
				print(f'Current model was generated in {model_dt.year}. {model_dt.month}. {model_dt.day}.')
				print('Updating model...')

			callbacks = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

			for start, end in update_queue:
				self.train_model_from_db(start, end, callbacks=callbacks, save=True)
			
			self.save_info(update_dt)


	def generate_model(self, input_size, output_size):
		'''ANN 모델 생성'''
		self.model = tf.keras.models.Sequential([
			tf.keras.Input(shape=(input_size,)),
			tf.keras.layers.Dense(50, activation='relu'),
			# tf.keras.layers.Dense(200, activation='relu'),
			tf.keras.layers.Dense(50, activation='relu'),
			tf.keras.layers.Dense(50, activation='relu'),
			tf.keras.layers.Dense(output_size)
		])


	def initialize_model(self, epochs=100):

		train_start = ti.train_start_dt(self.generated, self.bounds, self.cool)
		self.train_model_from_db(
				ti.datetime_to_tuple(train_start),
				ti.datetime_to_tuple(train_start+timedelta(hours=1)),
				epochs=epochs,
				save=True)
		self.update_model(self.generated, alert=False)


	def train_model(self, x_train, y_train, x_test=None, y_test=None, \
			epochs=100, callbacks=None, save=False):
		'''ANN을 훈련시킴'''

		self.model.compile(optimizer='adam',
						   loss='mse',
						   metrics=['mse'])

		self.scaler_x.fit(x_train)
		self.scaler_y.fit(y_train)
		
		x_train = self.scaler_x.transform(x_train)
		y_train = self.scaler_y.transform(y_train)

		# callback = tf.keras.callbacks.EarlyStopping(patience=5)
		self.history = self.model.fit(x_train, y_train, epochs=epochs,
			validation_split=0.0, shuffle=True, callbacks=callbacks)

		if x_test is not None:
			x_test = self.scaler_x.transform(x_test)
			y_test = self.scaler_y.transform(y_test)

			self.model.evaluate(x_test,  y_test, verbose=2)

		if save:
			self.save_model()


	def run_model(self, input_vector):
		'''ANN으로 출력변수 예측'''
		
		try:
			input_vector[0][0]
		except (TypeError, IndexError):
			input_vector = [input_vector]
		input_vector_scaled = self.scaler_x.transform(input_vector)
		predict_vector = self.model(input_vector_scaled)
		output_vector = self.scaler_y.inverse_transform(predict_vector)

		return output_vector


	#? ======== Database Related =======================
	#?  (Only for training, not for operation)

	def generate_dataset(self, time1, time2):

		db = self.db

		time_dt1 = datetime(*time1)
		time_dt2 = datetime(*time2)

		dataset_len = int((time_dt2 - time_dt1).total_seconds() / (60*15))

		input_dataset = np.zeros((dataset_len, self.input_size))
		label_dataset = np.zeros((dataset_len, self.output_size))

		sub_db = db.retrieve_from_db(
			[l[1] for l in self.input_past] + self.input_control + self.output,
			ti.datetime_to_tuple(time_dt1 - self.TS[0]*ti.TS),
			ti.datetime_to_tuple(time_dt2 + self.TS[1]*ti.TS),
		)

		for idx in range(dataset_len):

			time = ti.datetime_to_tuple(time_dt1 + ti.TS*idx)

			input_vector = []
			label_vector = []

			for ts, names in self.input_past:
				input_vector += list(db.get_sequence(names, time, -ts, table=sub_db))

			for names in self.input_control:
				if names[1] == '냉열원 기동상태':
					control_scheme_new = db.get_sequence(names, time, self.TS[1], table=sub_db) \
							* db.get_sequence(self.input_control[0], time, self.TS[1], table=sub_db)
					# print(control_scheme)
					# print(list(control_scheme_new))
					input_vector += list(control_scheme_new)
				else:
					input_vector += list(db.get_sequence(names, time, self.TS[1], table=sub_db))

			for names in self.output:
				label_vector += list(db.get_sequence(names, time, self.TS[1], table=sub_db))

			input_dataset[idx, :] = input_vector
			label_dataset[idx, :] = label_vector

		# if self.output_size == 1:
		#     label_dataset = np.array(label_dataset)

		print('====')
		print(f'Input Dataset : {input_dataset.shape}')
		print(f'Label Dataset : {label_dataset.shape}')
		print('====')

		return input_dataset, label_dataset


	def train_model_from_db(self, time1, time2, epochs=100, callbacks=None, save=False):

		x_train, y_train = self.generate_dataset(time1, time2)
		self.train_model(x_train, y_train, epochs=epochs, callbacks=callbacks, save=save)

		self.save_info(datetime(*time2))


	def test_model_from_db(self, time1, time2):
		
		x_test, y_test = self.generate_dataset(time1, time2)

		return y_test, self.run_model(x_test)


	#? ======== Data Management ========================


	#? ======== Predictions & Controls =================

	def predict_with_control(self, time, control_schemes, table=None):

		db = self.db
		
		if len(control_schemes) != len(self.input_control):
			raise ValueError(f'Need control scheme for {len(self.input_control)} '
				+ f'variables, but got {len(control_schemes)}.')
		for idx, control_scheme in enumerate(control_schemes):
			if len(control_scheme) != self.TS[1]:
				raise ValueError(f'Control scheme {idx} should be {self.TS[1]} '
					+ f'long, not {len(control_scheme)}.')

		if isinstance(table, pd.DataFrame):
			sub_db = table
		else:
			sub_db = db.retrieve_from_db(
				[l[1] for l in self.input_past],
				time,
				-max([l[0] for l in self.input_past]),
			)

		input_vector = []

		for ts, names in self.input_past:
			input_vector += list(db.get_sequence(names, time, -ts, table=sub_db))

		for names, control_scheme in zip(self.input_control, control_schemes):
			# input_vector += control_scheme
			
			if names[1] == '냉열원 기동상태' or names[1] == '온열원 기동상태':
				control_scheme_new = np.array(control_scheme) * np.array(control_schemes[0])
				# print(control_scheme)
				# print(list(control_scheme_new))
				input_vector += list(control_scheme_new)
			else:
				input_vector += control_scheme
			

		output_vector = self.run_model(input_vector)

		return output_vector


class HDC_AHU_COOL(HDC_AHU_COMPONENT):

	TS = (8, 4*4)  # timestep (past, future)

	MAX_TARGET_TS = 4*3
	MIN_ON_TARGET_TS = 4

	input_past = [
		(TS[0], ('AHU', '환기건구온도')),
		(TS[0], ('백엽상', '외기건구온도')),
		(TS[0], ('백엽상', '외기상대습도')),
		# (TS[0], ('AHU', '전열교환기 운전상태')),
		# (TS[0], ('AHU', 'IAQ 현재 풍량(회기풍량)')),
		# (TS[0], ('AHU', 'IAQ 현재 댐퍼개도')),
		# (TS[0], ('AHU', '급기팬 인버터 주파수')),
		# (TS[0], ('AHU', '환기팬 인버터 주파수')),
		# (TS[0], ('AHU', '코일 냉수 계산유량')),
		(TS[0], ('AHU', '냉방 급기팬 계산전력량')),
		(TS[0], ('AHU', '공조기 기동상태')),
		(TS[0], ('AHU', '냉수 코일입구온도')),
	]
	input_control = [
		('AHU', '공조기 기동상태'),
		('AHU', '냉열원 기동상태'),
	]
	output = [
		('AHU', '환기건구온도'),
	]

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)


class HDC_AHU_HEAT(HDC_AHU_COMPONENT):

	TS = (8, 4*4)  # timestep (past, future)

	MAX_TARGET_TS = 4*3
	MIN_ON_TARGET_TS = 4

	input_past = [
		(TS[0], ('AHU', '환기건구온도')),
		(TS[0], ('백엽상', '외기건구온도')),
		(TS[0], ('백엽상', '외기상대습도')),
		# (TS[0], ('AHU', '전열교환기 운전상태')),
		# (TS[0], ('AHU', 'IAQ 현재 풍량(회기풍량)')),
		# (TS[0], ('AHU', 'IAQ 현재 댐퍼개도')),
		# (TS[0], ('AHU', '급기팬 인버터 주파수')),
		# (TS[0], ('AHU', '환기팬 인버터 주파수')),
		# (TS[0], ('AHU', '코일 냉수 계산유량')),
		# (TS[0], ('AHU', '난방 급기팬 계산전력량')),
		(TS[0], ('AHU', '공조기 기동상태')),
		(TS[0], ('AHU', '온수 코일입구온도')),
	]
	input_control = [
		('AHU', '공조기 기동상태'),
		('AHU', '온열원 기동상태'),
	]
	output = [
		('AHU', '환기건구온도'),
	]

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)


class HDC_IPARK_AHU_HEAT(HDC_AHU_COMPONENT):

	TS = (8, 4*4)  # timestep (past, future)

	MAX_TARGET_TS = 4*3
	MIN_ON_TARGET_TS = 4

	input_past = [
		(TS[0], ('AHU', 'AHU 환기(RA)온도')),
		(TS[0], ('외부', '외기온도')),
		(TS[0], ('외부', '외기습도')),
		(TS[0], ('AHU', '급기팬 운전상태')),
		(TS[0], ('냉온수기1', '냉온수기1-1 출구온도')),
		# (TS[0], ('AHU', '코일 온도 x 밸브 개도율')),
	]
	input_control = [
		('AHU', '급기팬 운전상태'),
		('AHU', '온열원 기동상태'),
	]
	output = [
		('AHU', 'AHU 환기(RA)온도'),
	]

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)


class HDC_AHU:

	def __init__(self,
			source='111',
			save_loc='Models/', name=None,
			bounds=((5,11), (10,25)),
			now='now', db=None,
			refresh_cool=False, refresh_heat=False, refresh=False):

		if refresh:
			refresh_cool = True
			refresh_heat = True
		
		self.source = '공조기_AHU-' + source[-3:]

		if name:
			self.name = name
		else:
			self.name = 'ahu_' + source[-3:]
		
		self.model_cool = HDC_AHU_COOL(True, self.source, save_loc, self.name+'_cool',
				bounds, now=now, db=db, refresh=refresh_cool)
		self.model_heat = HDC_AHU_HEAT(False, self.source, save_loc, self.name+'_heat',
				bounds, now=now, db=db, refresh=refresh_heat)


	def update(self, now='now'):
		self.model_cool.update_model(now)
		self.model_heat.update_model(now)


	def opt_ctrl(self, start_time, is_start, target_ts, target_temp, \
			is_cool, sch, margin=0.15, sub_db=None, update=False):

		time = start_time

		if is_cool:
			model = self.model_cool
		else:
			model = self.model_heat

		db = model.db

		if update:
			model.update_model(datetime(*start_time))

		time_0 = ti.datetime_to_tuple(datetime(*time) + ti.TS)


		result = {}

		if is_start:
			if is_cool: BUFFER_TS = 4
			else: BUFFER_TS = 2
		else: BUFFER_TS = 1
		
		if model.TS[1] < model.MIN_ON_TARGET_TS + BUFFER_TS:
			raise ValueError('>> Wrong buffer timestep value.')


		outputs = {}

		curr_time = None
		success = False
		table = None

		if is_cool:
			#? 냉방
			if is_start:
				#? 기동 제어
				# for on_ts in range(target_ts-self.MIN_ON_TARGET_TS_COOL, 0, -1):
				for on_ts in range(target_ts, 0, -1):
					curr_time = ti.datetime_to_tuple(datetime(*time) + ti.TS*on_ts)

					ahu_ctrl_scheme = [0]*(on_ts-1) + [1]*(model.TS[1]-on_ts+1)
					cool_ctrl_scheme = sch
					control_schemes = [ahu_ctrl_scheme, cool_ctrl_scheme]
					# print(datetime(*time) + ti.TS*on_ts)
					output_0 = db.get_sequence(model.output[0], time, 1, table=sub_db)
					output_raw = model.predict_with_control(time_0, control_schemes, table=sub_db)[0]
					output = list(output_0) + list(output_raw)
					# outputs[f'{curr_time[3]}:{curr_time[4]:02}'] = output
					outputs[ti.time_to_string(*curr_time)] = output

					time_idx = ti.time_idx_range(time, 1+model.TS[1])
					# table = pd.DataFrame(
					#     {
					#         'AHU OP': [0]+ahu_ctrl_scheme,
					#         'COOL OP': [0]+cool_ctrl_scheme,
					#         'RA': output,
					#     },
					#     index=time_idx
					# )

					success = sum(output[target_ts: target_ts+BUFFER_TS]) \
						/ BUFFER_TS < target_temp + margin

					if success:
						break
				else:
					success = False
			
			else:
				#? 정지 제어
				for off_ts in range(1, target_ts+1):
					curr_time = ti.datetime_to_tuple(datetime(*time) + ti.TS*off_ts)

					ahu_ctrl_scheme = [1]*(off_ts-1) + [0]*(model.TS[1]-off_ts+1)
					cool_ctrl_scheme = sch
					control_schemes = [ahu_ctrl_scheme, cool_ctrl_scheme]
					# print(datetime(*time) + ti.TS*off_ts)
					output_0 = db.get_sequence(model.output[0], time, 1, table=sub_db)
					output_raw = model.predict_with_control(time_0, control_schemes, table=sub_db)[0]
					output = list(output_0) + list(output_raw)
					# outputs[f'{curr_time[3]}:{curr_time[4]:02}'] = output
					outputs[ti.time_to_string(*curr_time)] = output

					diff = [output[_+1]-output[_] for _ in range(len(output)-1)]
					is_flat = [_ < min(0.15, (max(diff))/2) for _ in diff]
					# print((max(output)-min(output))/6)
					# is_flat = [_ < 0.15 for _ in diff]
					for indicator_idx in range(off_ts-1, len(is_flat)):
						if is_flat[indicator_idx]: break
					indicator_idx += 1
					indi = [1 if _==indicator_idx else 0 for _ in range(len(is_flat))]
					

					time_idx = ti.time_idx_range(time, 1+model.TS[1])
					# table = pd.DataFrame(
					#     {
					#         'AHU OP': [1]+ahu_ctrl_scheme,
					#         'COOL OP': [1]+cool_ctrl_scheme,
					#         # 'Diff': [None]+diff,
					#         # 'Flat': [None]+is_flat,
					#         # 'Test': [0]+indi,
					#         'RA': output,
					#     },
					#     index=time_idx
					# )

					success = (indicator_idx >= target_ts) \
						& (sum(output[target_ts-BUFFER_TS: target_ts+BUFFER_TS]) \
						/ BUFFER_TS/2 < target_temp + margin)

					if success:
						break
				else:
					success = False
		
		else:
			#? 난방
			if is_start:
				#? 기동 제어
				# for on_ts in range(target_ts-self.MIN_ON_TARGET_TS_COOL, 0, -1):
				for on_ts in range(target_ts, 0, -1):
					curr_time = ti.datetime_to_tuple(datetime(*time) + ti.TS*on_ts)

					ahu_ctrl_scheme = [0]*(on_ts-1) + [1]*(model.TS[1]-on_ts+1)
					cool_ctrl_scheme = sch
					control_schemes = [ahu_ctrl_scheme, cool_ctrl_scheme]
					# print(datetime(*time) + ti.TS*on_ts)
					output_0 = db.get_sequence(model.output[0], time, 1, table=sub_db)
					output_raw = model.predict_with_control(time_0, control_schemes, table=sub_db)[0]
					output = list(output_0) + list(output_raw)
					# outputs[f'{curr_time[3]}:{curr_time[4]:02}'] = output
					outputs[ti.time_to_string(*curr_time)] = output

					time_idx = ti.time_idx_range(time, 1+model.TS[1])
					# table = pd.DataFrame(
					#     {
					#         'AHU OP': [0]+ahu_ctrl_scheme,
					#         'BOIL OP': [0]+cool_ctrl_scheme,
					#         'RA': output,
					#     },
					#     index=time_idx
					# )

					success = sum(output[target_ts: target_ts+BUFFER_TS]) \
						/ BUFFER_TS > target_temp - margin

					if success:
						break
				else:
					success = False
			
			else:
				#? 정지 제어
				for off_ts in range(1, target_ts+1):
					curr_time = ti.datetime_to_tuple(datetime(*time) + ti.TS*off_ts)

					ahu_ctrl_scheme = [1]*(off_ts-1) + [0]*(model.TS[1]-off_ts+1)
					cool_ctrl_scheme = sch
					control_schemes = [ahu_ctrl_scheme, cool_ctrl_scheme]
					# print(datetime(*time) + ti.TS*off_ts)
					output_0 = db.get_sequence(model.output[0], time, 1, table=sub_db)
					output_raw = model.predict_with_control(time_0, control_schemes, table=sub_db)[0]
					output = list(output_0) + list(output_raw)
					# outputs[f'{curr_time[3]}:{curr_time[4]:02}'] = output
					outputs[ti.time_to_string(*curr_time)] = output

					diff = [output[_+1]-output[_] for _ in range(len(output)-1)]
					is_flat = [_ < min(0.15, (max(diff))/2) for _ in diff]
					# print((max(output)-min(output))/6)
					# is_flat = [_ < 0.15 for _ in diff]
					for indicator_idx in range(off_ts-1, len(is_flat)):
						if is_flat[indicator_idx]: break
					indicator_idx += 1
					indi = [1 if _==indicator_idx else 0 for _ in range(len(is_flat))]
					

					time_idx = ti.time_idx_range(time, 1+model.TS[1])
					# table = pd.DataFrame(
					#     {
					#         'AHU OP': [1]+ahu_ctrl_scheme,
					#         'BOIL OP': [1]+cool_ctrl_scheme,
					#         # 'Diff': [None]+diff,
					#         # 'Flat': [None]+is_flat,
					#         # 'Test': [0]+indi,
					#         'RA': output,
					#     },
					#     index=time_idx
					# )

					success = (indicator_idx >= target_ts) \
						& (sum(output[target_ts-BUFFER_TS: target_ts+BUFFER_TS]) \
						/ BUFFER_TS/2 < target_temp + margin)
					success = sum(output[target_ts: target_ts+BUFFER_TS]) \
						/ BUFFER_TS > target_temp + margin

					if success:
						break
				else:
					success = False
		
		table = pd.DataFrame(
			{
				'AHU OP': list(db.get_sequence(
						model.input_control[0], time_0, -model.TS[0], table=sub_db)) \
						+ ahu_ctrl_scheme,
				('COOL OP' if is_cool else 'BOIL OP'): list(db.get_sequence(
						('Complex', model.input_control[1][1]), time_0, -model.TS[0], table=sub_db)) \
						+ sch,
				'RA': list(db.get_sequence(model.output[0], time_0, -model.TS[0], table=sub_db)) \
						+ list(output_raw)
			},
			index=ti.time_idx_range(time_0, -model.TS[0]) + ti.time_idx_range(time_0, model.TS[1])
		)
		# table.rename(columns={'OP': ('COOL OP' if is_cool else 'BOIL OP')})
		# print(db.retrieve_from_db(model.output, ti.datetime_to_tuple(datetime(*time_0)-ti.TS*model.TS[0]), model.TS[0]+model.TS[1]))

		summary = '\n'.join([
			'\n==== Control Summary ========\n',
			' < 설정 >',
			'  - 현재 시간     : ' + ti.time_to_string(*time),
			'  - 기동 / 정지   : ' + '기동' if is_start else '정지',
			'  - 목표 timestep : ' + str(target_ts),
			'  - 목표 온도     : ' + str(target_temp),
			'  - 냉방 / 난방   : ' + '냉방' if is_cool else '난방',
			'',
			' < 결과 >',
			'  - '+('기동' if is_start else '정지')+' 시점     : ' + ti.time_to_string(*curr_time),
			'  - 성공 / 실패   : ' + '성공' if success else '실패',
			'\n=============================\n'
		])

		result['params'] = dict(
			start_time  = start_time,
			is_start    = is_start,
			target_ts   = target_ts,
			target_temp = target_temp,
			is_cool     = is_cool,
			sch         = sch,
			margin      = margin,
		)
		result['start time'] = start_time
		result['time'] = curr_time
		result['success'] = success
		result['summary'] = summary
		result['table'] = table
		# result['temp'] = temperature
		result['target temp'] = target_temp
		result['df'] = pd.DataFrame(outputs, index=time_idx)

		return result


	def optimal_control(self, start_time, is_start, target, target_temp, \
			is_cool, sch, margin=0.15, update=False):
		if is_cool:
			model = self.model_cool
		else:
			model = self.model_heat

		res_file = f'{model.save_loc}{model.name}_optctrl_v3.pkl'


		if isinstance(start_time, tuple):
			start_time_raw = datetime(*start_time)
		elif isinstance(start_time, datetime):
			start_time_raw = start_time
		elif isinstance(start_time, str) and start_time.lower() == 'now':
			start_time_raw = datetime.now()

		start_time_aligned = datetime(
			start_time_raw.year,
			start_time_raw.month,
			start_time_raw.day,
			start_time_raw.hour,
			(start_time_raw.minute//15)*15,
		)
		if start_time_raw == start_time_aligned:
			start_time_dt = start_time_aligned - ti.TS
		else:
			start_time_dt = start_time_aligned
		start_time = ti.datetime_to_tuple(start_time_dt)


		if isinstance(target, int):
			target_ts = target
		else:
			if isinstance(target, datetime):
				target_dt = target
			elif isinstance(target, tuple) or isinstance(target, list):
				target_dt = datetime(*target)
			#? 목표 타임스텝 계산
			if target_dt < start_time_dt:
				raise ValueError('목표 시간은 시작 시간 이후여야 합니다.')
			else:
				target_ts = int((target_dt-start_time_dt)/ti.TS)


		#* 제어 시간 검사
		start_hour = start_time_dt.hour
		start_min = start_time_dt.minute
		override_settings = False

		if is_start:
			#? 기동 제어
			#* 시작 시간 관련
			if start_hour < 5:
				print('기동 제어 예측 시작 시간은 05:00 ~ 08:00 여야 입니다.')
				override_settings = True

				new_start_time_dt = datetime(*start_time[:3], 15)-timedelta(days=1)
				while new_start_time_dt.weekday() > 4:
					new_start_time_dt -= timedelta(days=1)
				
				is_start = False
			
			elif start_hour > 8 or (start_hour == 8 and start_min > 0):
				print('기동 제어 예측 시작 시간은 05:00 ~ 08:00 여야 입니다.')
				override_settings = True

				new_start_time_dt = datetime(*start_time[:3], 6)

		else:
			#? 정지 제어
			#* 시작 시간 관련
			if start_hour < 14:
				print('정지 제어 예측 시작 시간은 14:00 ~ 18:00 여야 입니다.')
				override_settings = True

				new_start_time_dt = datetime(*start_time[:3], 6)

				is_start = True
			
			elif start_hour > 18 or (start_hour == 18 and start_min > 0):
				print('정지 제어 예측 시작 시간은 14:00 ~ 18:00 여야 입니다.')
				override_settings = True

				new_start_time_dt = datetime(*start_time[:3], 15)
			
		allowed_target = [
			start_time_dt + model.MIN_ON_TARGET_TS*ti.TS,
			start_time_dt + model.MAX_TARGET_TS*ti.TS
		]
		

		if override_settings:
			#? 시작 시간 조건 충족 못할 지, 기본 설정으로 과거 제어 진행

			start_time = ti.datetime_to_tuple(new_start_time_dt)
			start_time_dt = new_start_time_dt
			
			if os.path.isfile(res_file):
				with open(res_file, 'rb') as f:
					last_res = pickle.load(f)

				last_start_time_dt = datetime(*last_res['start time'])
				if abs(last_start_time_dt - start_time_dt) < timedelta(hours=5):
					#? 마지막 제어 시작이 새 시작 시간과 5시간 이내로 차이날 경우
					print('마지막 제어 결과를 출력합니다.')
					return last_res
			
			
			print('마지막 제어 결과가 없습니다. 기본 설정으로 진행합니다.')

			target_ts = 12

			if is_start:
				if is_cool:
					target_temp = 24
				else:
					target_temp = 26
			else:
				if is_cool:
					target_temp = 27
				else:
					target_temp = 26

			sch = model.db.get_sequence(
				('Complex', model.input_control[1][1]),
				(*start_time[:4], 15),
				(*start_time[:3], start_time[3]+int(model.TS[1]/4), 15)
			)
			
			target_time_dt = start_time_dt + ti.TS*target_ts
			print(f' - 공조기 최적 {("기동" if is_start else "정지")} 제어 ' \
					+ f'({("냉방" if is_cool else "난방")})')
			print(f' - 목표 온도 : {target_temp}°C')
			print(f' - 시작 시간 : ' \
					+ f'{start_time_dt.year}년 ' \
					+ f'{start_time_dt.month}월 ' \
					+ f'{start_time_dt.day}일 ' \
					+ f'{start_time_dt.hour}시')
			print(f' - 목표 시간 : ' \
					+ f'{target_time_dt.year}년 ' \
					+ f'{target_time_dt.month}월 ' \
					+ f'{target_time_dt.day}일 ' \
					+ f'{target_time_dt.hour}시')

		'''
		#* 목표 시간 관련
		if is_start:
			#? 기동 제어
			if target_ts < model.MIN_ON_TARGET_TS or target_ts > model.MAX_TARGET_TS:
				raise ValueError(
					'가능한 목표 시간은 ' \
					+ f'{allowed_target[0].hour:02}:{allowed_target[0].minute:02} ~ ' \
					+ f'{allowed_target[1].hour:02}:{allowed_target[1].minute:02} ' \
					+ '입니다.'
				)
		else:
			#? 정지 제어
			if target_ts > model.MAX_TARGET_TS:
				raise ValueError(
					'가능한 목표 시간은 ~ ' \
					f'{allowed_target[1].hour:02}:{allowed_target[1].minute:02} 입니다.'
				)
		'''

		#* 실제 제어
		sch = list(sch)

		sub_db = model.db.retrieve_from_db(
			[l[1] for l in model.input_past] + model.input_control + model.output,
			ti.datetime_to_tuple(datetime(*start_time)+ti.TS),
			- max([l[0] for l in model.input_past]) - 1,
		)

		if is_start:
			#? 기동 제어
			if is_cool:
				#? 냉방

				if target_temp < 22 or target_temp > 27:
					raise ValueError('Target temperature must be 22<= and <=27.')
				
				target_temp_list = list(np.arange(target_temp, 27.5, 0.5))
				if len(target_temp_list) == 0: target_temp_list = [target_temp]
				for target_temp_new in target_temp_list:
					res = self.opt_ctrl(start_time, is_start, target_ts, target_temp_new, \
							is_cool, sch, margin, sub_db=sub_db, update=update)
					if res['success']: break
			
			else:
				#? 난방
				target_temp_list = list(np.arange(target_temp, 23.5, -0.5))
				if len(target_temp_list) == 0: target_temp_list = [target_temp]
				for target_temp_new in target_temp_list:
					res = self.opt_ctrl(start_time, is_start, target_ts, target_temp_new, \
							is_cool, sch, margin, sub_db=sub_db, update=update)
					if res['success']: break
		else:
			#? 정지 제어
			if is_cool:
				#? 냉방
				
				if target_temp < 22 or target_temp > 28:
					raise ValueError('Target temperature must be 22<= and <=28.')
				if start_time[3] < 14:
					raise ValueError('Starting time should be earlier than 8:00')

			
			res = self.opt_ctrl(start_time, is_start, target_ts, target_temp, \
					is_cool, sch, margin, sub_db=sub_db, update=update)

		with open(res_file, 'wb') as f:
			pickle.dump(res, f)

		#plot_ctrl_result(model, res)

		return res


class HDC_IPARK_AHU(HDC_AHU):

	def __init__(self,
			source='지상3층',
			save_loc='Models/', name=None,
			bounds=((5,11), (10,22)),
			now='now', db=None,
			refresh_cool=False, refresh_heat=False, refresh=False):

		if refresh:
			refresh_cool = True
			refresh_heat = True

		ahu_key = re.match(r'지([상하])(\d*)층', source)
		# ahu_loc = f'{"" if ahu_key[2]=="상" else "B"}{ahu_key[3]}F'
		ahu_loc = int(ahu_key[2])
		if ahu_loc == 2: ahu_size = 4
		else: ahu_size = 6

		if ahu_key[1] == '하' or ahu_loc not in range(2,10):
			raise ValueError('해당 공조기는 사용할 수 없습니다.')
		
		self.source = f'AHU{ahu_size} (지상{ahu_loc}층)'

		if name:
			self.name = name
		else:
			self.name = f'ahu_ipark_{ahu_loc}F'

		self.ahu_size = ahu_size
		self.ahu_loc = ahu_loc
		
		self.model_cool = None
		self.model_heat = HDC_IPARK_AHU_HEAT(False, self.source, save_loc, self.name+'_heat',
				bounds, now=now, db=db, refresh=refresh_heat)

