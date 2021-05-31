import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import MinMaxScaler
import sys;sys.path.append('..')
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import joblib 
import psychrolib as psy
import datetime 
from RealTimeDataAccumulator  import get_weather_forecast
import os
from HDC_XL import HDC_IPARK_XL
import pickle

def main_EC(currentDateTime, InputParam_EC, AHUidx_EC):
	os.chdir(r'D:\IPTower\EnthalpyControl\\')

	# %% 사용자 입력 변수
	"1. 사용자 입력 변수"
	#입력 변수 #1
	ahu_name = str(InputParam_EC[AHUidx_EC]['AHU'])
	print('ahu_name',ahu_name)
	#ahu_name=str("AHU6 (지상4층)")

	#입력 변수 #2
	real_time = 0 #실시간 OFF/ON [OFF:0,ON:1]

	#입력 변수 #3
	past_year = 2020
	past_month = 12
	past_today = 10
	past_hour = 14
	past_minute = 0

	#입력 변수 #4
	# 전열 교환기가 있으면 해당 데이터를 불러오고, 없으면 전열 교환기 데이터를 0으로 변환하여 모델 입력 변수에 적용
	ahu_heat_ex = 0 # 전열교환기 X/O[0, 1]
	if ahu_heat_ex == 1:
		ahu_heat_ex_base = ahu_name
	elif ahu_heat_ex ==0:
		ahu_heat_ex_base = str("AHU6 (지상3층)")
		
	# #엔탈피 제어 기간 오류 로직
	# if past_month < 4 or past_month >10:
	#     raise ValueError('엔탈피 제어 수행 기간이 아닙니다.')

	# 모델 경로 탐색
	model_path = './Ipark_model'
	if not os.path.exists(model_path):
		os.mkdir(model_path)
	# %% 모델 업데이트 기준
	"2. 모델 실시간 업데이트 기준"
	# 현재 시간
	if real_time == 1:  
		now = datetime.datetime.now()
		year = int(now.strftime('%y'))+2000
		month = int(now.strftime('%m'))
		month = str(month)
		today = int(now.strftime('%d'))
		today=str(today)
		hour = int(now.strftime('%H'))
		minute = int(now.strftime('%M'))
		# 15분 간격 에러 방지
		if minute == 0:
			minute = 58
			hour = hour-1
		elif minute == 15:
			minute = minute-2
		elif minute == 30:
			minute = minute-2
		elif minute == 45:
			minute = minute-2
			
	# 과거 시간
	else:     
		minute=0
		if past_minute <15:
			minute=0
		elif past_minute>14 and past_minute<30:
			minute=15
		elif past_minute>29 and past_minute<45:
			minute=30
		elif past_minute>44 and past_minute<60:
			minute=45
		year = past_year
		month = past_month
		month = str(month)
		today = past_today
		today=str(today)
		hour = past_hour
		minute = minute
		
	# 모델 저장 이름
	model_raen_date="./Ipark_model/"+month+today+"_"+ahu_name+"_model_raen.h5"
	model_ma_date="./Ipark_model/"+month+today+"_"+ahu_name+"_model_ma.h5"
	model_ma_date_base="./Ipark_model/"+month+today+"_AHU6 (지상3층)"+"_model_ma.h5"

	# 모델 저장 날짜
	date_check = './Ipark_model/' + month + today
	if os.path.exists(date_check): # 가동시간 체크용 파일이 존재하는지 확인
		# 존재할 경우 : 비교대상 로드 및 체크 후 업데이트 여부 결정
		with open(date_check,'rb') as check_time:
			date_check_load = pickle.load(check_time) # 비교대상 날짜 로드
	else:
		for file in os.scandir('./Ipark_model'):
			os.remove(file.path)
		with open(date_check,'wb') as check_time:
			pickle.dump(date_check, check_time) # 비교대상 날짜 저장
			
	# 환기정보 모델 로드
	if os.path.exists(model_raen_date):
		case_raen=1
	else:
		case_raen=2
		
	# 혼합온도 모델 로드
	if os.path.exists(model_ma_date):
		case_ma=1
	else:
		case_ma=2

	# 데이터 불러오기를 위한 정수 변환
	month=int(month)
	today=int(today)
	# %% HDC 데이터 불러오기
	"3. HDC 데이터 불러오기"
	# 기상 데이터 및 AHU111 댐퍼개도율 불러오기
	weather_station=str("외부")
	ahu_damper_base=str("AHU6 (지상3층)")
	ach_name=str("냉온수기1") # 냉온수기
	#모델 구축 시간
	"추후 실제 사용시 4월로 수정"
	train_month=10
	train_day=10

	# SQL 연결
	ipark_xl = HDC_IPARK_XL(refresh=False)

	# 외기 냉방 적용 주요 시점인, 4월 1일부터 1일씩 누적하여 모델 훈련
	## 백엽상 데이터
	sub_db_1_1 = ipark_xl.retrieve_from_db(
		[
			('외부', '외기온도'),
			('외부', '외기습도'),
		],
		(year, train_month, train_day, 0,0), (year, month, today,0,0),
	)
		
	sub_db_2_1 = ipark_xl.retrieve_from_db(
		[
			 (ahu_name, 'AHU 급기(SA)온도'), #0
			 (ahu_name, 'AHU 환기(RA)온도'), #1
			 (ahu_name, 'AHU 환기(RA)습도'), #2
			 (ahu_name, 'AHU 혼합(MA)온도'), #3
			 (ach_name, '냉온수기1-1 출구온도'), #4 냉수 코일입구온도 -> 냉온수기1-1 출구온도
			 (ach_name, '냉온수기 입구온도'), #5 냉수 코일출구온도 -> 냉온수기 입구온도
			 (ahu_name, '냉수 코일 밸브 개도율'), #6 냉수 유량 -> 코일 밸브 개도율
			 (ahu_name, 'AHU 외기(OA)댐퍼개도율'), #7
			 (ahu_name, '급기팬 운전상태'), #8 급기팬인버터주파수 -> 급기팬 운전상태
			 # (ahu_name, '전열교환기 운전상태'), #9 없음
		],
			(year, train_month, train_day, 0, 0), (year, month, today, 0, 0))
		
	## 결측치 제거
	sub_db_1_1_pad=sub_db_1_1.fillna(method='pad')
	sub_db_2_1_pad=sub_db_2_1.fillna(method='pad')

	weather_db_train=sub_db_1_1_pad.to_numpy()
	ahu_db_train=sub_db_2_1_pad.to_numpy()

	## 훈련 데이터 불러오기 설정 함수
	ahu_sa_t_train = ahu_db_train[:,0].reshape((-1,1)) # 급기온도
	ahu_ra_t_train= ahu_db_train[:,1].reshape((-1,1)) # 환기건구온도
	ahu_ra_h_train = ahu_db_train[:,2].reshape((-1,1))/100 # 환기상대습도
	ahu_ma_t_train = ahu_db_train[:,3].reshape((-1,1))  # 혼합온도
	ahu_coil_st_train = ahu_db_train[:,4].reshape((-1,1)) # 냉온수기1-1 출구온도
	ahu_coil_rt_train = ahu_db_train[:,5].reshape((-1,1)) # 냉온수기 입구온도
	ahu_coil_f_train = ahu_db_train[:,6].reshape((-1,1)) # 코일 밸브 개도율
	ahu_damper_train = ahu_db_train[:,7].reshape((-1,1)) # 외기 댐퍼 개도율
	ahu_sa_inv_train = ahu_db_train[:,8].reshape((-1,1)) # 급기팬 운전 상태
	# ahu_heat_ex_train = ahu_db_train[:,9].reshape((-1,1)) # 전열 교환기 데이터 없음
	# 전열 교환기 0 데이터 생성
	if ahu_heat_ex == 0:
		lens_train = len(ahu_sa_inv_train)
		ahu_heat_ex_train = np.zeros(lens_train).reshape(lens_train,1)
		
	ahu_oa_t_train = weather_db_train[:,0].reshape((-1,1)) # 외기온도
	ahu_oa_rh_train = weather_db_train[:,1].reshape((-1,1))/100 #외기상대습도

	## 훈련 데이터 길이
	num_low = len(ahu_oa_t_train)

	## 습공기선도(실내/실외 엔탈피를 구하기 위한 코드)
	psy.SetUnitSystem(psy.SI)

	## OA humidity ratio (절대습도)
	OA_hr_train = np.zeros((num_low,1))
	for i in range(1):
		for j in range(num_low):
			OA_hr_train[j,i] = psy.GetHumRatioFromRelHum(ahu_oa_t_train[j,i], ahu_oa_rh_train[j,i], 101325)

	## RA humidity ratio (절대습도)
	RA_hr_train = np.zeros((num_low,1))
	for i in range(1):
		for j in range(num_low):
			RA_hr_train[j,i] = psy.GetHumRatioFromRelHum(ahu_ra_t_train[j,i], ahu_ra_h_train[j,i], 101325)

	## OA enthalpy (단위 J/kg)
	OA_en_train = np.zeros((num_low,1))
	for i in range(1):
		for j in range(num_low):
			OA_en_train[j,i] = psy.GetMoistAirEnthalpy(ahu_oa_t_train[j,i], OA_hr_train[j,i])

	## RA enthalpy (단위 J/kg)
	RA_en_train = np.zeros((num_low,1))
	for i in range(1):
		for j in range(num_low):
			RA_en_train[j,i] = psy.GetMoistAirEnthalpy(ahu_ra_t_train[j,i], RA_hr_train[j,i])
			
	## 훈련 데이터 절대습도/엔탈피
	ahu_oa_hr_train = OA_hr_train
	ahu_oa_en_train = OA_en_train
	ahu_ra_hr_train = RA_hr_train
	ahu_ra_en_train= RA_en_train
	# %% 환기 정보 예측 모델
	"4. 환기 정보 예측 모델 구현"
	"훈련 날짜가 현재 날짜보다 앞설 경우, 에러가 발생함"
	TS_PAST, TS_FUTURE = 3*4, 1*4  # 과거, 미래의 timestep 개수(시간*4(15분 4개:1시간))
	input_vector_num = len(ahu_db_train) - (TS_PAST + TS_FUTURE) #데이터셋 행의 전체 길이-과거와 미래 열 길이
	input_vector_size = TS_PAST * 7 + TS_FUTURE #(TS_PAST: 과거 15분 간격 *4=1시간, 2시간인 경우, 2시간*4획) 열의 길이
	input_dataset = np.zeros((input_vector_num, input_vector_size)) # 입력변수 데이터셋 생성
	label_dataset = np.zeros((input_vector_num, TS_FUTURE*2)) # 출력변수 데이터셋 생성

	# 입력 변수 Time-shift
	for idx in range(input_vector_num):
		idx_now = idx + TS_PAST
		input_vector = list(ahu_coil_st_train[idx: idx+TS_PAST]) # 냉온수기1-1 출구온도
		input_vector += list(ahu_coil_rt_train[idx: idx+TS_PAST]) # 냉온수기 입구온도
		input_vector += list(ahu_coil_f_train[idx: idx+TS_PAST]) # 코일 밸브 개도율
		input_vector += list(ahu_sa_t_train[idx: idx+TS_PAST]) # 급기온도
		input_vector += list(ahu_sa_inv_train[idx: idx+TS_PAST]) # 급기팬 운전 상태
		input_vector += list(ahu_ra_en_train[idx: idx+TS_PAST]) # 환기 엔탈피
		input_vector += list(ahu_ra_t_train[idx: idx+TS_PAST])  # 환기온도
		input_vector += list(ahu_oa_t_train[idx_now: idx_now+TS_FUTURE])  #외기 정보
		input_dataset[idx, :] = input_vector # 입력 변수 데이터셋
		
	# 출력 변수 Time-shift
		label_vector = list(ahu_ra_t_train[idx_now: idx_now+TS_FUTURE]) # 환기온도
		label_vector += list(ahu_ra_en_train[idx_now: idx_now+TS_FUTURE]) # 환기엔탈피
		label_dataset[idx, :] = label_vector # 출력 변수 데이터셋

	all_number=len(ahu_db_train) 
	train_number=all_number

	# Train 기간 정의
	train_size = train_number - TS_PAST
	x_train_raen, y_train_raen = input_dataset[:train_size, :], label_dataset[:train_size, :]

	# 정규화 스케일러 저장
	scaler_x_raen, scaler_y_raen = MinMaxScaler(), MinMaxScaler()
	scaler_x_raen.fit(x_train_raen)
	scaler_y_raen.fit(y_train_raen)

	x_train_raen = scaler_x_raen.transform(x_train_raen)
	y_train_raen = scaler_y_raen.transform(y_train_raen)

	os.chdir(r'D:\IPTower\EnthalpyControl\\')
	joblib.dump(scaler_x_raen, './Ipark_model/' + ahu_name + '_scaler_x_raen.gz')
	joblib.dump(scaler_y_raen, './Ipark_model/' + ahu_name + '_scaler_y_raen.gz')

	#모델 파라미터 설정
	x_train_batch_size = 100 # 배치 사이즈 결정
	epochs=300 # 모델 epoch 설정

	# 모델 업데이트 
	## 저장된 모델의 날짜가 같으면 불러오기
	if case_raen==1:
		model_raen = tf.keras.models.load_model(model_raen_date)
		
	## 저장된 모델의 날짜가 다르면 모델 생성
	elif case_raen ==2:

		# ANN 모델 구축
		# 파라미터 설정

		model_raen=keras.models.Sequential([keras.Input(input_vector_size),
									keras.layers.Dense(30),
									keras.layers.BatchNormalization(),
									keras.layers.Dense(30, activation='relu'),
									keras.layers.BatchNormalization(),
									keras.layers.Dense(30, activation='relu'),
									keras.layers.BatchNormalization(),
									keras.layers.Dense(30, activation='relu'),
									keras.layers.Dense(TS_FUTURE*2)])
		model_raen.compile(loss='mse', optimizer='adam', metrics="MeanAbsoluteError")
		
		model_raen.fit(x_train_raen, y_train_raen, x_train_batch_size, epochs)
		model_raen.save(model_raen_date)
		
	# % 훈련 데이터 예측
	y_train_prediction_sc_raen=model_raen.predict(x_train_raen) # 훈련 기간 출력 변수 예측
	y_train_prediction_sc_raen=y_train_prediction_sc_raen[:,:] # array 배열
	y_train_prediction_raen = scaler_y_raen.inverse_transform(y_train_prediction_sc_raen) # 훈련 기간 출력 변수 예측 값: 정규화 상수 -> 일반 상수 transform
	y_train_prediction_raen=y_train_prediction_raen[:,:] # 훈련 기간 출력 변수 예측 값
	y_train_raen = scaler_y_raen.inverse_transform(y_train_raen) # 훈련 기간 출력 변수 실측 값: 정규화 상수 -> 일반 상수 transform

	# 훈련 데이터 t+1 기준 정확도
	##환기온도 예측 정확도
	y_prediction_t1_raen=y_train_prediction_raen[:,0] 
	y_train_t1_raen=y_train_raen[:,0] 
	y_RMSE_t1_raen = np.sqrt(np.sum((y_prediction_t1_raen-y_train_t1_raen)**2)/len(y_prediction_t1_raen))
	y_CVRMSE_t1_raen = y_RMSE_t1_raen/np.mean(y_prediction_t1_raen)*100
	y_MBE_t1_raen = np.sum(y_prediction_t1_raen-y_train_t1_raen)/sum(y_prediction_t1_raen)*100

	##환기엔탈피 예측 정확도
	y_prediction_t2_raen=y_train_prediction_raen[:,4] # 8:00~18:00기준
	y_train_t2_raen=y_train_raen[:,4] # 8:00~18:00기준
	y_RMSE_t2_raen = np.sqrt(np.sum((y_prediction_t2_raen-y_train_t2_raen)**2)/len(y_prediction_t2_raen))
	y_CVRMSE_t2_raen = y_RMSE_t2_raen/np.mean(y_prediction_t2_raen)*100
	y_MBE_t2_raen = np.sum(y_prediction_t2_raen-y_train_t2_raen)/sum(y_prediction_t2_raen)*100
	print("y_MBE_t1_ra=", y_MBE_t1_raen, "y_CVRMSE_t1_ra=",y_CVRMSE_t1_raen)
	print("y_MBE_t2_en=", y_MBE_t2_raen, "y_CVRMSE_t2_en=",y_CVRMSE_t2_raen)
	# %% 실시간 환기정보 예측
	"5. 실시간 환기정보 예측"
	# 과거 훈련 시간 정의
	## 입력변수에 포함되는 과거 데이터 indexing
	## 과거 15분 이하 일때, 타임 테이블이 분으로 안불러와지는 문제(14분 -> -1) 해결
	if minute<15:
		hour_under=hour-4
	elif minute>14:
		hour_under=hour-3
	if minute<15:
		minute_under=minute+60-15
	elif minute>14:
		minute_under=minute-15   
		
	# SQL 서버 연결 

	sub_db_1_2 = ipark_xl.retrieve_from_db(
		[
			('외부', '외기온도'),
			('외부', '외기습도'),
		],
			(year, month, today, hour_under, minute_under), (year,month, today, hour, minute)).drop_duplicates()

	sub_db_2_2 =ipark_xl.retrieve_from_db(
		[
			 (ahu_name, 'AHU 급기(SA)온도'), #0
			 (ahu_name, 'AHU 환기(RA)온도'), #1
			 (ahu_name, 'AHU 환기(RA)습도'), #2
			 (ahu_name, 'AHU 혼합(MA)온도'), #3
			 (ach_name, '냉온수기1-1 출구온도'), #4 냉수 코일입구온도 -> 냉온수기1-1 출구온도
			 (ach_name, '냉온수기 입구온도'), #5 냉수 코일출구온도 -> 냉온수기 입구온도
			 (ahu_name, '냉수 코일 밸브 개도율'), #6 냉수 유량 -> 코일 밸브 개도율
			 (ahu_name, 'AHU 외기(OA)댐퍼개도율'), #7
			 (ahu_name, '급기팬 운전상태'), #8 급기팬인버터주파수 -> 급기팬 운전상태
			 # (ahu_name, '전열교환기 운전상태'), #9! 없음
		],

			(year, month, today, hour_under,  minute_under), (year,month, today, hour, minute)).drop_duplicates()


	sub_db_1_2_pad=sub_db_1_2.fillna(method='bfill')
	sub_db_2_2_pad=sub_db_2_2.fillna(method='bfill')


	weather_db_test=sub_db_1_2_pad.to_numpy()
	ahu_db_test=sub_db_2_2_pad.to_numpy()

	# 행 길이 맞추기 
	if len(weather_db_test)==13 or len(ahu_db_test)==13:
		weather_db_test=copy.deepcopy(weather_db_test[1:,:])
		ahu_db_test=copy.deepcopy(ahu_db_test[1:,:])

	# % 향후 1시간 예측을 위한 초기 데이터 불러오기
	ahu_sa_t_test = ahu_db_test[:,0].reshape((-1,1)) #급기온도
	ahu_ra_t_test= ahu_db_test[:,1].reshape((-1,1)) # 환기온도
	ahu_ra_h_test = ahu_db_test[:,2].reshape((-1,1))/100 # 환기상대습도
	ahu_ma_t_test = ahu_db_test[:,3].reshape((-1,1)) # 혼합온도
	ahu_coil_st_test = ahu_db_test[:,4].reshape((-1,1)) #냉온수기1-1 출구온도
	ahu_coil_rt_test = ahu_db_test[:,5].reshape((-1,1)) #냉온수기 입구온도
	ahu_coil_f_test = ahu_db_test[:,6].reshape((-1,1)) #코일 밸브 개도율
	ahu_damper_test = ahu_db_test[:,7].reshape((-1,1)) #외기 댐퍼 개도율
	ahu_sa_inv_test = ahu_db_test[:,8].reshape((-1,1)) #급기팬 운전 상태
	# ahu_heat_ex_test = ahu_db_test[:,9].reshape((-1,1)) #전열교환기 운전상태 데이터 없음
	# 전열 교환기 0 데이터 생성
	if ahu_heat_ex == 0:
		lens_test = len(ahu_sa_inv_test)
		ahu_heat_ex_test = np.zeros(lens_test).reshape(lens_test,1)
		
	ahu_oa_t_test = weather_db_test[:,0].reshape((-1,1)) # 외기온도
	ahu_oa_rh_test = weather_db_test[:,1].reshape((-1,1))/100 #외기상대습도

	num_low = len(ahu_oa_t_test)

	# %습공기선도(실내/실외 엔탈피를 구하기 위한 코드)
	psy.SetUnitSystem(psy.SI)

	## OA humidity ratio (절대습도)
	OA_hr_test = np.zeros((num_low,1))
	for i in range(1):
		for j in range(num_low):
			OA_hr_test[j,i] = psy.GetHumRatioFromRelHum(ahu_oa_t_test[j,i], ahu_oa_rh_test[j,i], 101325)

	# RA humidity ratio (절대습도)
	RA_hr_test = np.zeros((num_low,1))
	for i in range(1):
		for j in range(num_low):
			RA_hr_test[j,i] = psy.GetHumRatioFromRelHum(ahu_ra_t_test[j,i], ahu_ra_h_test[j,i], 101325)

	## OA enthalpy (단위 J/kg)
	OA_en_test = np.zeros((num_low,1))
	for i in range(1):
		for j in range(num_low):
			OA_en_test[j,i] = psy.GetMoistAirEnthalpy(ahu_oa_t_test[j,i], OA_hr_test[j,i])

	## RA enthalpy (단위 J/kg)
	RA_en_test = np.zeros((num_low,1))
	for i in range(1):
		for j in range(num_low):
			RA_en_test[j,i] = psy.GetMoistAirEnthalpy(ahu_ra_t_test[j,i], RA_hr_test[j,i])
			
	## 실시간 데이터 절대습도/엔탈피
	ahu_oa_hr_test = OA_hr_test
	ahu_oa_en_test = OA_en_test
	ahu_ra_hr_test = RA_hr_test
	ahu_ra_en_test= RA_en_test 

	## Data: Time-shift 
	TS_PAST , TS_FUTURE = 3*4, 1*4  # 과거, 미래의 timestep 개수(시간*4(15분 4개:1시간))
	input_vector_num = 1 #1시간
	input_vector_size = TS_PAST * 7 + TS_FUTURE #(TS_PAST: 과거 15분 간격 *4=1시간, 2시간인 경우, 2시간*4획) 열의 길이
	input_dataset = np.zeros((input_vector_num, input_vector_size)) # 입력변수 데이터셋 생성

	"5-1 현재 시간 기준 실시간 시뮬레이터 "
	if real_time==1:

		# 기상API 연결:향후 1시간 외기온도 계산
		## 아이파크타워
		nx = str(61) 			# 예보지점 x 좌표
		ny = str(126) 			# 예보지점 y 좌표
		
		## HDC측 기상 API 로드
		temp_wf, humid_wf = get_weather_forecast(nx, ny) 
		## 현재 기준 가장 가까운 미래 시간 불러오기
		today_hour_time_p = temp_wf[1][3] ; today_hour_time_p = int(today_hour_time_p)/100
		## 미래 시간: 외기온도
		today_hour_temp_p = temp_wf[1][4] ; today_hour_temp_p = int(today_hour_temp_p)
		## 미래 시간: 외기습도
		today_hour_humid_p = humid_wf[1][4] ; today_hour_humid_p = int(today_hour_humid_p)
		## 향후 1시간 외기온도
		temp_past=weather_db_test[11][0]
		temp_future=today_hour_temp_p # 예측값
		temp_future_h=np.linspace(temp_past,temp_future,4)
		## 향후 1시간 외기 상대습도
		humid_past=weather_db_test[11][1]
		humid_future=today_hour_humid_p
		humid_future_h=np.linspace(humid_past,humid_future,4).reshape((1,4)) 
		
		for idx in range(input_vector_num):
			idx_now = idx + TS_PAST
			input_vector = list(ahu_coil_st_test[idx: idx+TS_PAST]) # 냉온수기1-1 출구온도
			input_vector += list(ahu_coil_rt_test[idx: idx+TS_PAST]) # 냉온수기 입구온도
			input_vector += list(ahu_coil_f_test[idx: idx+TS_PAST]) # 코일 밸브 개도율
			input_vector += list(ahu_sa_t_test[idx: idx+TS_PAST]) # 급기 온도
			input_vector += list(ahu_sa_inv_test[idx: idx+TS_PAST]) # 급기팬 운전상태
			input_vector += list(ahu_ra_en_test[idx: idx+TS_PAST]) # 환기 엔탈피
			input_vector += list(ahu_ra_t_test[idx: idx+TS_PAST])  # 환기온도
			input_vector += list(temp_future_h)  #외기 정보
			input_dataset[idx, :] = input_vector # 입력 변수 데이터셋

	# 실시간이 아닐 경우
		"5-2 과거 시간 기준 실시간 시뮬레이터 "
	else:        
		sub_db_1_1h_f = ipark_xl.retrieve_from_db(
		[
			('외부', '외기온도'),
			('외부', '외기습도'),
		],
				(year, month, today, hour, minute), (year, month, today, hour+1, minute))
		sub_db_1_1h_f_pad=sub_db_1_1h_f.fillna(method='bfill')
		sub_db_1_1h_f=sub_db_1_1h_f_pad.to_numpy()
		# 행 길이 맞추기 
		# % 향후 1시간 예측을 위한 초기 데이터 불러오기
		ahu_oa_t_future = sub_db_1_1h_f[:,0].reshape((-1,1)) # 외기온도
		
		for idx in range(input_vector_num):
			idx_now = idx + TS_PAST
			input_vector = list(ahu_coil_st_test[idx: idx+TS_PAST]) # 냉온수기1-1 출구온도
			input_vector += list(ahu_coil_rt_test[idx: idx+TS_PAST])  # 냉온수기 입구온도
			input_vector += list(ahu_coil_f_test[idx: idx+TS_PAST])  # 코일 밸브 개도율
			input_vector += list(ahu_sa_t_test[idx: idx+TS_PAST]) # 급기 온도
			input_vector += list(ahu_sa_inv_test[idx: idx+TS_PAST])  # 급기팬 운전상태
			input_vector += list(ahu_ra_en_test[idx: idx+TS_PAST]) # 환기 엔탈피
			input_vector += list(ahu_ra_t_test[idx: idx+TS_PAST])  # 환기온도
			input_vector += list(ahu_oa_t_future[idx: TS_FUTURE])  #외기 정보
			input_dataset[idx, :] = input_vector # 입력 변수 데이터셋

	# 입력 변수 Time-shift
	x_test_raen = input_dataset
	x_test_raen = scaler_x_raen.transform(x_test_raen)

	# 스케일러 저장
	joblib.dump(scaler_x_raen, './ipark_model/' + ahu_name + '_scaler_x_raen.gz')
	joblib.dump(scaler_y_raen, './ipark_model/' + ahu_name + '_scaler_y_raen.gz')
	# 실시간 예측
	y_prediction_sc_raen=model_raen.predict(x_test_raen) 
	y_prediction_sc_raen=y_prediction_sc_raen[:,:] 
	y_prediction_raen = scaler_y_raen.inverse_transform(y_prediction_sc_raen)
	y_prediction_raen=y_prediction_raen[:,:] #array 배열
	# 향후 1시간 환기온도 예측
	y_prediction_ra=y_prediction_raen[:,:4]
	#향후 1시간 환기엔탈피 예측
	y_prediction_en=y_prediction_raen[:,4:]
	# %%  혼합온도 예측 MPC 모델
	"6. 혼합온도 예측 MPC 모델 구현"
	# MPC, 혼합 엔탈피 모델 타임스텝 정의
	all_number=len(ahu_db_train) #train 24:00분까지
	train_number=all_number
	TS_PAST, TS_FUTURE = 1*1, 1*4  # 과거, 미래의 timestep 개수(시간*4(15분 4개:1시간))
	train_size = train_number - TS_PAST # 상수:엑셀 내 훈련데이터 범위, 상수: 엑셀 index, 실제 데
	input_vector_num = len(ahu_db_train) - (TS_PAST + TS_FUTURE) #데이터셋 행의 전체 길이-과거와 미래 열 길이
	input_vector_size = TS_PAST * 3 + TS_FUTURE *3 #(TS_PAST: 과거 15분 간격 *4=1시간, 2시간인 경우, 2시간*4획) 열의 길이
	input_dataset = np.zeros((input_vector_num, input_vector_size)) # 입력변수 데이터셋 생성
	label_dataset = np.zeros((input_vector_num, TS_FUTURE)) # 출력변수 데이터셋 생성

	# 입력 변수 Time-shift
	for idx in range(input_vector_num):
		idx_now = idx + TS_PAST
		input_vector = list(ahu_oa_t_train[idx: idx+TS_PAST]) #외기온도(과거)
		input_vector += list(ahu_ra_t_train[idx: idx+TS_PAST]) # 환기온도(과거)
		input_vector += list(ahu_heat_ex_train[idx: idx+TS_PAST]) # 환기온도(과거)
		input_vector += list(ahu_oa_t_train[idx_now: idx_now+TS_FUTURE]) # 외기온도(미래)
		input_vector += list(ahu_ra_t_train[idx_now: idx_now+TS_FUTURE]) # 환기온도(미래)
		input_vector += list(ahu_damper_train[idx_now: idx_now+TS_FUTURE])  # 외기 댐퍼 개도율
		input_dataset[idx, :] = input_vector
		
	# 출력 변수 Time-shift
		label_vector = list(ahu_ma_t_train[idx_now: idx_now+TS_FUTURE]) # 혼합온도
		label_dataset[idx, :] = label_vector 
		
	# Train 기간 정의 
	x_train_ma, y_train_ma = input_dataset[:train_size, :], label_dataset[:train_size, :]#(24시 00분까지 INDEX)

	# 정규화 스케일러 저장
	scaler_x_ma, scaler_y_ma = MinMaxScaler(), MinMaxScaler()
	scaler_x_ma.fit(x_train_ma)
	scaler_y_ma.fit(y_train_ma)

	x_train_ma_sc = scaler_x_ma.transform(x_train_ma)
	y_train_ma_sc = scaler_y_ma.transform(y_train_ma)

	joblib.dump(scaler_x_ma, './Ipark_model/' + ahu_name + '_scaler_x_ma.gz')
	joblib.dump(scaler_y_ma, './Ipark_model/' + ahu_name + '_scaler_y_ma.gz')

	# ANN 모델 구축
	if case_ma==1:
		model_ma = tf.keras.models.load_model(model_ma_date)
	elif case_ma==2:
		model_ma=keras.models.Sequential([keras.Input(input_vector_size),
									keras.layers.Dense(30),
									keras.layers.BatchNormalization(),
									keras.layers.Dense(30, activation='relu'),
									keras.layers.BatchNormalization(),
									keras.layers.Dense(30, activation='relu'),
									keras.layers.BatchNormalization(),
									keras.layers.Dense(30, activation='relu'),
									keras.layers.Dense(TS_FUTURE)])
		model_ma.compile(loss='mse', optimizer='adam', metrics="MeanAbsoluteError")
		model_ma.fit(x_train_ma_sc, y_train_ma_sc, x_train_batch_size, epochs)

	# % 모델 저장
		model_ma.save(model_ma_date)
		y_train_prediction_sc_ma=model_ma.predict(x_train_ma_sc) # 훈련 기간 출력 변수 예측
		y_train_prediction_sc_ma=y_train_prediction_sc_ma[:,:] # array 배열
		y_train_prediction_ma = scaler_y_ma.inverse_transform(y_train_prediction_sc_ma) # 훈련 기간 출력 변수 예측 값: 정규화 상수 -> 일반 상수 transform
		y_train_prediction_ma=y_train_prediction_ma[:,:] # 훈련 기간 출력 변수 예측 값
	   
		# % 훈련 데이터 정확도
		y_train_RMSE_ma = np.sqrt(np.sum((y_train_prediction_ma-y_train_ma)**2)/len(y_train_prediction_ma))
		y_train_CVRMSE_ma = y_train_RMSE_ma/np.mean(y_train_prediction_ma)*100
		y_train_MBE_ma = np.sum(y_train_prediction_ma-y_train_ma)/sum(y_train_prediction_ma)*100
		
		# 훈련 데이터 t+1 기준 정확도
		y_prediction_t1_ma=y_train_prediction_ma[:,0] 
		y_train_t1_ma=y_train_ma[:,0]
		y_RMSE_t1_ma = np.sqrt(np.sum((y_prediction_t1_ma-y_train_t1_ma)**2)/len(y_prediction_t1_ma))
		y_CVRMSE_t1_ma = y_RMSE_t1_ma/np.mean(y_prediction_t1_ma)*100
		y_MBE_t1_ma = np.sum(y_prediction_t1_ma-y_train_t1_ma)/sum(y_prediction_t1_ma)*100
		print("y_MBE_t1_ma=", y_MBE_t1_ma, "y_CVRMSE_t1_ma=",y_CVRMSE_t1_ma)
		"모든 공조기에 외기 댐퍼 개도율이 존재하므로 elif 문 삭제"    
		# elif ahu_name=="공조기_AHU-101":
		#     model_ma=keras.models.Sequential([keras.Input(input_vector_size),
		#                                 keras.layers.Dense(30),
		#                                 keras.layers.BatchNormalization(),
		#                                 keras.layers.Dense(30, activation='relu'),
		#                                 keras.layers.BatchNormalization(),
		#                                 keras.layers.Dense(30, activation='relu'),
		#                                 keras.layers.BatchNormalization(),
		#                                 keras.layers.Dense(30, activation='relu'),
		#                                 keras.layers.Dense(TS_FUTURE)])
		#     model_ma.compile(loss='mse', optimizer='adam', metrics="MeanAbsoluteError")
		#     model_ma.fit(x_train_ma_sc, y_train_ma_sc, x_train_batch_size, epochs)
		
		# # % 모델 저장
		#     model_ma.save(model_ma_date)
			
		#     y_train_prediction_sc_ma=model_ma.predict(x_train_ma_sc) # 훈련 기간 출력 변수 예측
		#     y_train_prediction_sc_ma=y_train_prediction_sc_ma[:,:] # array 배열
		#     y_train_prediction_ma = scaler_y_ma.inverse_transform(y_train_prediction_sc_ma) # 훈련 기간 출력 변수 예측 값: 정규화 상수 -> 일반 상수 transform
		#     y_train_prediction_ma=y_train_prediction_ma[:,:] # 훈련 기간 출력 변수 예측 값
		#     # % 훈련 데이터 정확도
		#     y_train_RMSE_ma = np.sqrt(np.sum((y_train_prediction_ma-y_train_ma)**2)/len(y_train_prediction_ma))
		#     y_train_CVRMSE_ma = y_train_RMSE_ma/np.mean(y_train_prediction_ma)*100
		#     y_train_MBE_ma = np.sum(y_train_prediction_ma-y_train_ma)/sum(y_train_prediction_ma)*100
			
		#     # 훈련 데이터 t+1 기준 정확도
		#     y_prediction_t1_ma=y_train_prediction_ma[:,0]
		#     y_train_t1_ma=y_train_ma[:,0]
		#     y_RMSE_t1_ma = np.sqrt(np.sum((y_prediction_t1_ma-y_train_t1_ma)**2)/len(y_prediction_t1_ma))
		#     y_CVRMSE_t1_ma = y_RMSE_t1_ma/np.mean(y_prediction_t1_ma)*100
		#     y_MBE_t1_ma = np.sum(y_prediction_t1_ma-y_train_t1_ma)/sum(y_prediction_t1_ma)*100
		#     print("y_MBE_t1_ma=", y_MBE_t1_ma, "y_CVRMSE_t1_ma=",y_CVRMSE_t1_ma)

	# 기존 AHU-111 모델 불러오기
		# else: model_ma = tf.keras.models.load_model(model_ma_date_base)
	# %% 실시간 혼합온도 예측
	"7-1 현재 시간 기준 실시간 시뮬레이터 "

	if real_time==1:
		#"과거 3시간 예측 위한 데이터 불러오기 "
		hour_under=0 
		minute_under=0 
		now = datetime.datetime.now()
		year = int(now.strftime('%y'))+2000
		month = int(now.strftime('%m'))
		today = int(now.strftime('%d'))
		hour = int(now.strftime('%H'))
		minute = int(now.strftime('%M'))
		## 입력변수에 포함되는 과거 데이터 indexing
		if minute<30:
			hour_under=hour-4
		elif minute>29:
			hour_under=hour-3
		if minute<30:
			minute_under_3h=minute+60-30
		elif minute>29:
			minute_under_3h=minute-30
		# SQL 서버 연결 
		sub_db_1_3h = ipark_xl.retrieve_from_db(
		[
			('외부', '외기온도'),
			('외부', '외기습도'),
		],
				(year, month, today, hour_under, minute_under_3h), (year,month, today, hour, minute)).drop_duplicates()
		sub_db_2_3h = ipark_xl.retrieve_from_db(
		[
			 (ahu_name, 'AHU 급기(SA)온도'), #0
			 (ahu_name, 'AHU 환기(RA)온도'), #1
			 (ahu_name, 'AHU 환기(RA)습도'), #2
			 (ahu_name, 'AHU 혼합(MA)온도'), #3
			 (ach_name, '냉온수기1-1 출구온도'),  #4 냉수 코일입구온도 -> 냉온수기1-1 출구온도
			 (ach_name, '냉온수기 입구온도'), #5 냉수 코일출구온도 -> 냉온수기 입구온도
			 (ahu_name, '냉수 코일 밸브 개도율'), #6! # 냉수 유량 -> 코일 밸브 개도율
			 (ahu_name, 'AHU 외기(OA)댐퍼개도율'), #7
			 (ahu_name, '급기팬 운전상태'), #8! 급기팬인버터주파수 -> 급기팬 운전상태
			 # (ahu_name, '전열교환기 운전상태'), #9 없음
		],
				(year, month, today, hour_under,  minute_under_3h), (year,month, today, hour, minute)).drop_duplicates()
		
		if len(sub_db_1_3h)>12:
			sub_db_1_3h=sub_db_1_3h[1:][:]
		if len(sub_db_2_3h)>12:
			sub_db_2_3h=sub_db_2_3h[1:][:]
		sub_db_1_3h_pad=sub_db_1_3h.fillna(method='bfill')
		sub_db_2_3h_pad=sub_db_2_3h.fillna(method='bfill')
		weather_db_test_3h=sub_db_1_3h_pad.to_numpy()
		ahu_db_test_3h=sub_db_2_3h_pad.to_numpy()
		
		# % 향후 1시간 예측을 위한 초기 데이터 불러오기
		ahu_sa_t_test_3h = ahu_db_test_3h[:,0].reshape((-1,1)) #급기건구온도
		ahu_ra_t_test_3h= ahu_db_test_3h[:,1].reshape((-1,1)) # 환기건구온도
		ahu_ra_h_test_3h = ahu_db_test_3h[:,2].reshape((-1,1))/100 # 환기상대습도
		ahu_ma_t_test_3h = ahu_db_test_3h[:,3].reshape((-1,1)) # 혼합온도
		ahu_coil_st_test_3h = ahu_db_test_3h[:,4].reshape((-1,1)) # 냉온수기1-1 출구온도
		ahu_coil_rt_test_3h = ahu_db_test_3h[:,5].reshape((-1,1)) # 냉온수기 입구온도
		ahu_coil_f_test_3h = ahu_db_test_3h[:,6].reshape((-1,1)) # 코일 밸브 개도율
		ahu_damper_test_3h = ahu_db_test_3h[:,7].reshape((-1,1)) # 외기 댐퍼 개도율
		ahu_sa_inv_test_3h = ahu_db_test_3h[:,8].reshape((-1,1)) # 급기팬 운전 상태
		# ahu_heat_ex_test_3h = ahu_db_test_3h[:,9].reshape((-1,1))  # 전열 교환기 데이터 없음
		lens_3h = len(ahu_sa_inv_test_3h)
		if ahu_heat_ex == 0:
			ahu_heat_test_ex_3h = np.zeros(lens_3h).reshape(lens_3h,1)

		ahu_oa_t_test_3h = weather_db_test_3h[:,0].reshape((-1,1)) # 외기온도
		ahu_oa_rh_test_3h = weather_db_test_3h[:,1].reshape((-1,1))/100 #외기상대습도
		# %
		"과거 3시간 예측"
		# MPC, 코일 모델 타임스텝 정의
		all_number_3h=len(ahu_db_test_3h) #train 24:00분까지
		TS_PAST, TS_FUTURE = 1*1, 1*4  # 과거, 미래의 timestep 개수(시간*4(15분 4개:1시간))
		test_3h_size = all_number_3h - TS_PAST # 상수:엑셀 내 훈련데이터 범위, 상수: 엑셀 index, 실제 데이터 개수 +1(24시 00분까지 INDEX)
		# % Data: Time-shift 
		input_vector_num= len(ahu_db_test_3h) - (TS_PAST + TS_FUTURE) #데이터셋 행의 전체 길이-과거와 미래 열 길이
		input_vector_size = TS_PAST * 3 + TS_FUTURE *3 #(TS_PAST: 과거 15분 간격 *4=1시간, 2시간인 경우, 2시간*4획) 열의 길이
		input_dataset = np.zeros((input_vector_num, input_vector_size)) # 입력변수 데이터셋 생성
		label_dataset = np.zeros((input_vector_num, TS_FUTURE)) # 출력변수 데이터셋 생성
		
		# 입력 변수 Time-shift
		for idx in range(input_vector_num):
			idx_now = idx + TS_PAST
			input_vector = list(ahu_oa_t_test_3h[idx: idx+TS_PAST]) #외기온도(과거)
			input_vector += list(ahu_ra_t_test_3h[idx: idx+TS_PAST]) # 환기온도(과거)
			input_vector += list(ahu_heat_test_ex_3h[idx: idx+TS_PAST]) # 환기온도(과거)
			input_vector += list(ahu_oa_t_test_3h[idx_now: idx_now+TS_FUTURE]) # 외기온도(미래)
			input_vector += list(ahu_ra_t_test_3h[idx_now: idx_now+TS_FUTURE]) # 환기온도(미래)
			input_vector += list(ahu_damper_test_3h[idx_now: idx_now+TS_FUTURE])  # 외기 댐퍼 개도율
			input_dataset[idx, :] = input_vector

		# 출력 변수 Time-shift
			label_vector = list(ahu_ma_t_test_3h[idx_now: idx_now+TS_FUTURE]) # 혼합온도
			label_dataset[idx, :] = label_vector 
			
		# Train 기간 정의 
		x_test_ma_3h, y_test_ma_3h = input_dataset[:test_3h_size, :], label_dataset[:test_3h_size, :]#(24시 00분까지 INDEX)
		
		# 정규화 스케일러 저장
		scaler_x_ma, scaler_y_ma = MinMaxScaler(), MinMaxScaler()
		scaler_x_ma.fit(x_test_ma_3h)
		scaler_y_ma.fit(y_test_ma_3h)
		
		x_test_ma_sc_3h = scaler_x_ma.transform(x_test_ma_3h)
		y_test_ma_sc_3h = scaler_y_ma.transform(y_test_ma_3h)
		
		joblib.dump(scaler_x_ma, './Ipark_model/' + ahu_name + '_scaler_x_ma.gz')
		joblib.dump(scaler_y_ma, './Ipark_model/' + ahu_name + '_scaler_y_ma.gz')
		
		y_test_prediction_ma_sc_3h=model_ma.predict(x_test_ma_sc_3h) # 훈련 기간 출력 변수 예측
		y_test_prediction_ma_sc_3h=y_test_prediction_ma_sc_3h[:,:] # array 배열
		y_test_prediction_ma_3h = scaler_y_ma.inverse_transform(y_test_prediction_ma_sc_3h) # 훈련 기간 출력 변수 예측 값: 정규화 상수 -> 일반 상수 transform
		y_test_prediction_ma_3h=y_test_prediction_ma_3h[:,:] # 훈련 기간 출력 변수 예측 값
		y_test_prediction_ma_3h_1=y_test_prediction_ma_3h[0,:].reshape(-1,1)
		y_test_prediction_ma_3h_2=y_test_prediction_ma_3h[4,:].reshape(-1,1)
		y_test_prediction_ma_3h_3=y_test_prediction_ma_3h[7,:].reshape(-1,1)
		y_test_prediction_ma_3h_plot=np.vstack([y_test_prediction_ma_3h_1,y_test_prediction_ma_3h_2,y_test_prediction_ma_3h_3])
		
		# %혼합온도 실시간 예측
		"4. 혼합온도 실시간 예측"
		all_number=len(ahu_db_test) #train 24:00분까지
		TS_PAST, TS_FUTURE = 1*1, 1*4  # 과거, 미래의 timestep 개수(시간*4(15분 4개:1시간))
		
		# % Data: Time-shift 
		input_vector_num = 2
		input_vector_size = TS_PAST * 3 + TS_FUTURE *3 #(TS_PAST: 과거 15분 간격 *4=1시간, 2시간인 경우, 2시간*4획) 열의 길이
		input_dataset = np.zeros((input_vector_num, input_vector_size)) # 입력변수 데이터셋 생성
		label_dataset = np.zeros((input_vector_num, TS_FUTURE)) # 출력변수 데이터셋 생성
		
		# 15분전 외기, 환기 온도 불러오기
		ahu_oa_t_t15m=ahu_oa_t_test[11,0]
		ahu_ra_t_t15m=ahu_ra_t_test[11,0]
		ahu_heat_ex_15m=ahu_heat_ex_test[11,0]

		temp_future_h=temp_future_h.reshape(1,4)
		ahu_damper = np.zeros((1, 4)) # 출력변수 데이터셋 생성
		y_prediction_ra=y_prediction_ra.reshape(1,4)
		x_test_ma_p_c = np.hstack([ahu_oa_t_t15m, ahu_ra_t_t15m,ahu_heat_ex_15m]).reshape(1,3) 
		x_test_ma_p_c = np.hstack([x_test_ma_p_c,temp_future_h,y_prediction_ra,ahu_damper])
		
		# 제어 로직
		ahu_damper_control=x_test_ma_p_c[:, 11:] #댐퍼 개도율 데이터 불러오기
		OA_t=temp_future_h.reshape(4,1) # 외기 온도
		OA_rh=humid_future_h.reshape(4,1)/100 #외기 상대습도
		num_low = len(OA_rh)
		
		# 습공기선도(실내/실외 엔탈피를 구하기 위한 코드)
		psy.SetUnitSystem(psy.SI)
		
		## OA humidity ratio (절대습도)
		OA_hr= np.zeros((num_low,1))
		for i in range(1):
			for j in range(num_low):
				OA_hr[j,i] = psy.GetHumRatioFromRelHum(OA_t[j,i], OA_rh[j,i], 101325)
		
		## OA enthalpy (단위 J/kg)
		OA_en = np.zeros((num_low,1))
		for i in range(1):
			for j in range(num_low):
				OA_en[j,i] = psy.GetMoistAirEnthalpy(OA_t[j,i], OA_hr[j,i])
		
		# 제어 시퀀스 적용
		y_prediction_en=y_prediction_en.reshape(4,1)
		y_prediction_ra=y_prediction_ra.reshape(4,1)
		ahu_damper_control= ahu_damper_control.reshape(4,1)
		ahu_damper_test_15m=float(ahu_damper_test[11,:])
		
		for i in range(1):
			for j in range(4):
				# 외기 엔탈피 < 환기 엔탈피 and 외기 건구온도 < 환기온도 and 외기 절대습도 <0.012:
				if  OA_en[j,i]<y_prediction_en[j,i] and OA_t[j,i]<y_prediction_ra[j,i] and OA_hr[j,i]<0.012: # 외기냉방#1: 0.012=ASHRAE 55 기준 절대습도 recommended Limit
					ahu_damper_control[j,i]=100 # 댐퍼 개도율 100%
				# 외기 엔탈피 > 환기 엔탈피 and 댐퍼 개도율 >40:
				elif OA_en[j,i]>y_prediction_en[j,i] and ahu_damper_control[j,i]>30: 
					ahu_damper_control[j,i]=30 #댐퍼 개도율 30%
				elif ahu_damper_control[j,i]==0:
					ahu_damper_control[j,i]=ahu_damper_test_15m
					
		# % MPC 모델 TEST 입력 값에 개도율 제어 변수 삽입 
		# 제어 운영에 대한 댐퍼개도율
		ahu_damper_control=ahu_damper_control.reshape(1,4)
		x_test_ma_p_c[:,11:]=ahu_damper_control[:,:] # x 테스트
		
		# 기본 운영에 대한 댐퍼개도율
		x_test_ma_p=copy.deepcopy(x_test_ma_p_c)
		ahu_damper_p=ahu_damper_test[8:,0].reshape(1,4)
		x_test_ma_p[:,11:]=ahu_damper_p
		
		# 스케일 변환
		x_test_ma_c_sc = scaler_x_ma.transform(x_test_ma_p_c) #제어 로직 댐퍼개도율 입력값
		x_test_ma_sc = scaler_x_ma.transform(x_test_ma_p) #기본 운영 댐퍼개도율 입력값
		
		#혼합온도 예측: 댐퍼개도율 제어에 따른 향후 1시간 값 삽입
		## TEST 제어에 따른 혼합온도 예측
		y_prediction_ma_c_sc_ma=model_ma.predict(x_test_ma_c_sc) # test 기간 출력 변수 예측
		y_prediction_ma_c_sc_ma=y_prediction_ma_c_sc_ma[:,:] # array 배열
		y_prediction_ma_c = scaler_y_ma.inverse_transform(y_prediction_ma_c_sc_ma) # # test 기간 출력 변수 실측 값: 정규화 상수 -> 일반 상수 transform
		
		#혼합온도 예측: 댐퍼개도율 과거 1시간 값을 기준으로 향후 1시간 값 삽입
		## TEST 예측에 따른 혼합온도 예측
		y_prediction_ma_p_sc=model_ma.predict(x_test_ma_sc) # test 기간 출력 변수 예측
		y_prediction_ma_p_sc=y_prediction_ma_p_sc[:,:] # array 배열
		y_prediction_ma_p = scaler_y_ma.inverse_transform(y_prediction_ma_p_sc) # # test 기간 출력 변수 실측 값: 정규화 상수 -> 일반 상수 transform
		# % 혼합 엔탈피 계산
		"혼합 엔탈피 계산"
		# 과거 3시간 혼합 엔탈피 계산
		oa_3h=copy.deepcopy(ahu_oa_t_test_3h)
		ra_3h=copy.deepcopy(ahu_ra_t_test_3h)
		ma_3h=copy.deepcopy(ahu_ma_t_test_3h)
		oa_en_3h=copy.deepcopy(ahu_oa_en_test)
		ra_en_3h=copy.deepcopy(ahu_ra_en_test)
		length_3h=12
		x=np.zeros(length_3h)
		ma_en_3h_bems=np.zeros(length_3h)
		ma_en_3h_p=np.zeros(length_3h)
		## 과거 3시간 BEMS 혼합 엔탈피
		for i in range(length_3h):
			x[i]=(ra_3h[i]-ma_3h[i])/(ma_3h[i]-oa_3h[i])
			ma_en_3h_bems[i]=(x[i]/(x[i]+1))*oa_en_3h[i]+(1/(x[i]+1))*ra_en_3h[i]
		## 과거 3시간 예측 혼합 엔탈피 
		for i in range(length_3h):
			x[i]=(ra_3h[i]-y_test_prediction_ma_3h_plot[i])/(y_test_prediction_ma_3h_plot[i]/oa_3h[i])
			ma_en_3h_p[i]=(x[i]/(x[i]+1))*oa_en_3h[i]+(1/(x[i]+1))*ra_en_3h[i]
			
		# 향후 1시간 혼합 엔탈피 계산
		oa_1h=copy.deepcopy(temp_future_h).reshape(4,1) #외기온도
		ra_1h=copy.deepcopy(y_prediction_ra) # 환기온도
		ma_1h_p=copy.deepcopy(y_prediction_ma_p).reshape(4,1) #예측 혼합온도
		ma_1h_c=copy.deepcopy(y_prediction_ma_c).reshape(4,1) #예측 혼합온도
		oa_en_1h=copy.deepcopy(OA_en) # 외기엔탈피
		ra_en_1h=copy.deepcopy(y_prediction_en) # 환기 엔탈피
		length=len(ma_1h_p)
		x=np.zeros(length)
		ma_en_1h_p=np.zeros(length)
		ma_en_1h_c=np.zeros(length)
		## 향후 1시간 예측 혼합 엔탈피
		for i in range(length):
			x[i]=(ra_1h[i]-ma_1h_p[i])/(ma_1h_p[i]-oa_1h[i])
			ma_en_1h_p[i]=(x[i]/(x[i]+1))*oa_en_1h[i]+(1/(x[i]+1))*ra_en_1h[i]
			
		## 향후 1시간 제어 혼합 엔탈피
		for i in range(length):
			x[i]=(ra_1h[i]-ma_1h_c[i])/(ma_1h_c[i]-oa_1h[i])
			ma_en_1h_c[i]=(x[i]/(x[i]+1))*oa_en_1h[i]+(1/(x[i]+1))*ra_en_1h[i]  

	# 과거 혼합온도 예측
		"7-2 과거 시간 기준 실시간 시뮬레이터 "
		
	else:
		hour_under=0 
		minute_under_3h=0 
		hour_future=hour+1
		minute_future=0 
		## 입력변수에 포함되는 과거 데이터 indexing
		## 과거 15분 이하 일때, 타임 테이블이 분으로 안불러와지는 문제(14분 -> -1) 해결
		if minute<30:
			hour_under=hour-4
		elif minute>29:
			hour_under=hour-3
		if minute<30:
			minute_under_3h=minute+60-45
		elif minute>29:
			minute_under_3h=minute-30
		if minute_under_3h==60:
			minute_under_3h=0 
		## 미래 데이터 불러오기를 위한 hour,minute_future
		if minute<30:
			minute_future=minute+30
		elif minute>29 and minute<45:
			minute_future=0
		elif minute>44 and minute<60:
			minute_future=15
		if minute_future==0 or minute_future<30:
		   hour_future=hour+2
		   
		   
		# SQL 서버 연결 
		sub_db_1_ma_past = ipark_xl.retrieve_from_db(
		[
			('외부', '외기온도'),
			('외부', '외기습도'),
		],
				(year, month, today, hour_under, minute_under_3h), (year,month, today, hour_future, minute_future))
		sub_db_2_ma_past = ipark_xl.retrieve_from_db(
		[
			 (ahu_name, 'AHU 급기(SA)온도'), #0
			 (ahu_name, 'AHU 환기(RA)온도'), #1
			 (ahu_name, 'AHU 환기(RA)습도'), #2
			 (ahu_name, 'AHU 혼합(MA)온도'), #3
			 (ach_name, '냉온수기1-1 출구온도'), #4 냉수 코일입구온도 -> 냉온수기1-1 출구온도
			 (ach_name, '냉온수기 입구온도'), #5 냉수 코일출구온도 -> 냉온수기 입구온도
			 (ahu_name, '냉수 코일 밸브 개도율'), #6 냉수 유량 -> 코일 밸브 개도율
			 (ahu_name, 'AHU 외기(OA)댐퍼개도율'), #7
			 (ahu_name, '급기팬 운전상태'), #8! 급기팬인버터주파수 -> 급기팬 운전상태
			 # (ahu_name, '전열교환기 운전상태'), #9 없음
		],
				(year, month, today, hour_under,  minute_under_3h), (year,month, today, hour_future, minute_future))

		sub_db_1_ma_past_pad=sub_db_1_ma_past.fillna(method='bfill')
		sub_db_2_ma_past_pad=sub_db_2_ma_past.fillna(method='bfill')
		weather_db_test_ma_past=sub_db_1_ma_past_pad.to_numpy()
		ahu_db_test_ma_past=sub_db_2_ma_past_pad.to_numpy()
		
		# % 향후 1시간 예측을 위한 초기 데이터 불러오기
		ahu_sa_t_test_ma_past = ahu_db_test_ma_past[:,0].reshape((-1,1)) #급기건구온도
		ahu_ra_t_test_ma_past= ahu_db_test_ma_past[:,1].reshape((-1,1)) # 환기건구온도
		ahu_ra_h_test_ma_past = ahu_db_test_ma_past[:,2].reshape((-1,1))/100 # 환기상대습도
		ahu_ma_t_test_ma_past = ahu_db_test_ma_past[:,3].reshape((-1,1)) # 혼합온도
		ahu_coil_st_test_ma_past = ahu_db_test_ma_past[:,4].reshape((-1,1)) # 냉온수기1-1 출구온도
		ahu_coil_rt_test_ma_past = ahu_db_test_ma_past[:,5].reshape((-1,1))  # 냉온수기 입구온도
		ahu_coil_f_test_ma_past = ahu_db_test_ma_past[:,6].reshape((-1,1)) # 코일 밸브 개도율
		ahu_damper_test_ma_past = ahu_db_test_ma_past[:,7].reshape((-1,1)) # 외기 댐퍼 개도율
		ahu_sa_inv_test_ma_past = ahu_db_test_ma_past[:,8].reshape((-1,1)) # 급기팬 운전 상태
		# ahu_heat_ex_test_ma_past = ahu_db_test_ma_past[:,9].reshape((-1,1)) # 전열 교환기 데이터 없음
		if ahu_heat_ex == 0:
			lens_past = len(ahu_sa_inv_test_ma_past)
			ahu_heat_ex_test_ma_past = np.zeros(lens_past).reshape(lens_past,1)
		
		ahu_oa_t_test_ma_past = weather_db_test_ma_past[:,0].reshape((-1,1)) # 외기온도
		ahu_oa_rh_test_ma_past = weather_db_test_ma_past[:,1].reshape((-1,1))/100 #외기상대습도


		#"과거 3시간 예측"
		# MPC, 코일 모델 타임스텝 정의
		all_number_3h=len(ahu_db_test_ma_past) #train 24:00분까지
		TS_PAST, TS_FUTURE = 1*1, 1*4  # 과거, 미래의 timestep 개수(시간*4(15분 4개:1시간))
		test_ma_past_size = all_number_3h - TS_PAST # 상수:엑셀 내 훈련데이터 범위, 상수: 엑셀 index, 실제 데이터 개수 +1(24시 00분까지 INDEX)
		
		# % Data: Time-shift 
		input_vector_num= len(ahu_db_test_ma_past) - (TS_PAST + TS_FUTURE) #데이터셋 행의 전체 길이-과거와 미래 열 길이
		input_vector_size = TS_PAST * 3 + TS_FUTURE *3 #(TS_PAST: 과거 15분 간격 *4=1시간, 2시간인 경우, 2시간*4획) 열의 길이
		input_dataset = np.zeros((input_vector_num, input_vector_size)) # 입력변수 데이터셋 생성
		label_dataset = np.zeros((input_vector_num, TS_FUTURE)) # 출력변수 데이터셋 생성
		
		# 입력 변수 Time-shift
		for idx in range(input_vector_num):
			idx_now = idx + TS_PAST
			input_vector = list(ahu_oa_t_test_ma_past[idx: idx+TS_PAST]) #외기온도(과거)
			input_vector += list(ahu_ra_t_test_ma_past[idx: idx+TS_PAST]) # 환기온도(과거)
			input_vector += list(ahu_heat_ex_test_ma_past[idx: idx+TS_PAST]) # 환기온도(과거)
			input_vector += list(ahu_oa_t_test_ma_past[idx_now: idx_now+TS_FUTURE]) # 외기온도(미래)
			input_vector += list(ahu_ra_t_test_ma_past[idx_now: idx_now+TS_FUTURE]) # 환기온도(미래)
			input_vector += list(ahu_damper_test_ma_past[idx_now: idx_now+TS_FUTURE])  # 외기 댐퍼 개도율
			input_dataset[idx, :] = input_vector
			
		# 출력 변수 Time-shift
			label_vector = list(ahu_ma_t_test_ma_past[idx_now: idx_now+TS_FUTURE]) # 혼합온도
			label_dataset[idx, :] = label_vector 
			
		# Train 기간 정의 
		x_test_ma_past, y_test_ma_past = input_dataset[2:15,:], label_dataset[2:15,:]
		
		# 정규화 스케일러 저장
		# scaler_x_ma, scaler_y_ma = MinMaxScaler(), MinMaxScaler()
		# scaler_x_ma.fit(x_test_ma_past)
		# scaler_y_ma.fit(y_test_ma_past)
		
		x_test_ma_sc_ma_past = scaler_x_ma.transform(x_test_ma_past)
		y_test_ma_sc_ma_past = scaler_y_ma.transform(y_test_ma_past)
		
		
		# joblib.dump(scaler_x_ma, 'scaler_x_ma.gz')
		# joblib.dump(scaler_y_ma, 'scaler_y_ma.gz')
		
		y_test_prediction_ma_sc_past=model_ma.predict(x_test_ma_sc_ma_past) # 훈련 기간 출력 변수 예측
		y_test_prediction_ma_sc_past=y_test_prediction_ma_sc_past[:,:] # array 배열
		y_test_prediction_ma_past = scaler_y_ma.inverse_transform(y_test_prediction_ma_sc_past) # 훈련 기간 출력 변수 예측 값: 정규화 상수 -> 일반 상수 transform
		y_test_prediction_ma_past=y_test_prediction_ma_past[:,:] # 훈련 기간 출력 변수 예측 값
		
		# 제어 로직
		ahu_damper_control=copy.deepcopy(ahu_damper_test_ma_past[15:19, :]) #댐퍼 개도율 데이터 불러오기
		OA_t=ahu_oa_t_test_ma_past[15:19, :] # 외기 온도
		OA_rh=ahu_oa_rh_test_ma_past[15:19, :] #외기 상대습도
		num_low = len(OA_rh)

		# 습공기선도(실내/실외 엔탈피를 구하기 위한 코드)
		psy.SetUnitSystem(psy.SI)
		
		## OA humidity ratio (절대습도)
		OA_hr= np.zeros((num_low,1))
		for i in range(1):
			for j in range(num_low):
				OA_hr[j,i] = psy.GetHumRatioFromRelHum(OA_t[j,i], OA_rh[j,i], 101325)
		
		## OA enthalpy (단위 J/kg)
		OA_en = np.zeros((num_low,1))
		for i in range(1):
			for j in range(num_low):
				OA_en[j,i] = psy.GetMoistAirEnthalpy(OA_t[j,i], OA_hr[j,i])

		# 제어 시퀀스 적용
		for i in range(1):
			for j in range(4):
				# 외기 엔탈피 < 환기 엔탈피 and 외기 건구온도 < 환기온도 and 외기 절대습도 <0.012:
				if  OA_en[j,i]<y_prediction_en.reshape(4,1)[j,i] and OA_t[j,i]<y_prediction_ra.reshape(4,1)[j,i] and OA_hr[j,i]<0.012: # 외기냉방#1: 0.012=ASHRAE 55 기준 절대습도 recommended Limit
					ahu_damper_control[j,i]=100; # 댐퍼 개도율 100%
					print("damper_opening(100%): ",ahu_damper_control[j,i])
				# 외기 엔탈피 > 환기 엔탈피 and 댐퍼 개도율 > 30:
				elif OA_en[j,i]>y_prediction_en.reshape(4,1)[j,i] and ahu_damper_control[j,i]>30: 
					ahu_damper_control[j,i]=30; #댐퍼 개도율 30%
					print("damper_opening(30%): ",ahu_damper_control[j,i])
				elif ahu_damper_control[j,i]==0:
					ahu_damper_control[j,i]=0;
					print("damper_opening(0%): ",ahu_damper_control[j,i])
				else:
					print("damper_opening(preservation)=",ahu_damper_control[j,i])



		# % MPC 모델 TEST 입력 값에 개도율 제어 변수 삽입 
		# 제어 운영에 대한 댐퍼개도율
		x_test_ma_past_c= copy.deepcopy(x_test_ma_past)
		ahu_damper_control=ahu_damper_control.reshape(1,4)
		x_test_ma_past_c[12,11:]=ahu_damper_control[:,:] # x 테스트
		
		# 기본 운영에 대한 댐퍼개도율
		x_test_ma_past=copy.deepcopy(x_test_ma_past)
		
		# 스케일 변환
		x_test_ma_c_sc = scaler_x_ma.transform(x_test_ma_past_c) #제어 로직 댐퍼개도율 입력값
		x_test_ma_sc = scaler_x_ma.transform(x_test_ma_past) #기본 운영 댐퍼개도율 입력값
		
		#혼합온도 예측: 댐퍼개도율 제어에 따른 향후 1시간 값 삽입
		## TEST 제어에 따른 혼합온도 예측
		y_prediction_ma_c_sc_ma=model_ma.predict(x_test_ma_c_sc) # test 기간 출력 변수 예측
		y_prediction_ma_c_sc_ma=y_prediction_ma_c_sc_ma[:,:] # array 배열
		y_prediction_ma_c = scaler_y_ma.inverse_transform(y_prediction_ma_c_sc_ma) # # test 기간 출력 변수 실측 값: 정규화 상수 -> 일반 상수 transform
		
		#혼합온도 예측: 댐퍼개도율 과거 1시간 값을 기준으로 향후 1시간 값 삽입
		## TEST 예측에 따른 혼합온도 예측
		y_prediction_ma_p_sc=model_ma.predict(x_test_ma_sc) # test 기간 출력 변수 예측
		y_prediction_ma_p_sc=y_prediction_ma_p_sc[:,:] # array 배열
		y_prediction_ma_p = scaler_y_ma.inverse_transform(y_prediction_ma_p_sc) # # test 기간 출력 변수 실측 값: 정규화 상수 -> 일반 상수 transform
		# % 혼합 엔탈피 계산
		"측정 데이터 외기 엔탈피, 환기 엔탈피 계산"
		# 과거 3시간+향후 1시간 혼합 엔탈피 계산
		oa_t_past=copy.deepcopy(ahu_oa_t_test_ma_past[3:19])
		ra_t_past=copy.deepcopy(ahu_ra_t_test_ma_past[3:19])
		ma_t_past=copy.deepcopy(ahu_ma_t_test_ma_past[3:19])
		oa_rh_past=copy.deepcopy(ahu_oa_rh_test_ma_past[3:19])
		ra_rh_past=copy.deepcopy(ahu_ra_h_test_ma_past[3:19])
		# 환기습도 
		length_ma_past=len(oa_t_past)
		x=np.zeros(length_ma_past)
		ma_en_past_bems=np.zeros(length_ma_past)
		ma_en_past_p=np.zeros(length_ma_past)   
		
		# 습공기선도(실내/실외 엔탈피를 구하기 위한 코드)
		psy.SetUnitSystem(psy.SI)
		
		OA_hr_past = np.zeros((length_ma_past,1))
		for i in range(1):
			for j in range(length_ma_past):
				OA_hr_past[j,i] = psy.GetHumRatioFromRelHum(oa_t_past[j,i], oa_rh_past[j,i], 101325)
		
		# RA humidity ratio (절대습도)
		RA_hr_past = np.zeros((length_ma_past,1))
		for i in range(1):
			for j in range(length_ma_past):
				RA_hr_past[j,i] = psy.GetHumRatioFromRelHum(ra_t_past[j,i], ra_rh_past[j,i], 101325)
		
		## OA enthalpy (단위 J/kg)
		OA_en_past = np.zeros((length_ma_past,1))
		for i in range(1):
			for j in range(length_ma_past):
				OA_en_past[j,i] = psy.GetMoistAirEnthalpy(oa_t_past[j,i], OA_hr_past[j,i])
		
		## RA enthalpy (단위 J/kg)
		RA_en_past = np.zeros((length_ma_past,1))
		for i in range(1):
			for j in range(length_ma_past):
				RA_en_past[j,i] = psy.GetMoistAirEnthalpy(ra_t_past[j,i], RA_hr_past[j,i])
	   
		#"측정 데이터 혼합 엔탈피 계산"    
		# 측정데이터: 과거 3시간+미래 1시간 측정 데이터 혼합 엔탈피 계산
		ma_en_bems=np.zeros(length_ma_past)
		for i in range(length_ma_past):
			x[i]=(ra_t_past[i]-ma_t_past[i])/(ma_t_past[i]-oa_t_past[i])
			ma_en_bems[i]=(x[i]/(x[i]+1))*OA_en_past[i]+(1/(x[i]+1))*RA_en_past[i]

		#"예측 데이터 혼합 엔탈피 계산"
		# 모델 예측 데이터: 과거 3시간 예측+ 미래 1시간 예측 데이터 혼합 엔탈피 계산
		ma_en_p=np.zeros(length_ma_past)
		y_prediction_ma_3h=y_prediction_ma_p[:12,0].reshape(12,1) #과거 3시간 예측 값
		y_prediction_ma_1h=y_prediction_ma_p[12,:].reshape(4,1) # 미래 1시간 예측 값
		y_prediction_ma_plot=np.vstack([y_prediction_ma_3h,y_prediction_ma_1h]) #합치기
		# 과거 3시간 환기 온도 및 엔탈피 BEMS 값
		y_measured_ra_3h=ahu_ra_t_test_ma_past[3:15,:]
		y_measured_en_3h= RA_en_past[3:15,:]
		# 향후 1시간 환기 온도 및 엔탈피 예측 값
		ra_prediction_1h=copy.deepcopy(y_prediction_ra).reshape(4,1) # 환기온도
		ra_en_prediction_1h=copy.deepcopy(y_prediction_en).reshape(4,1) # 환기 엔탈피
		# 과거 3시간+ 향후 1시간 환기 온도 및 엔탈피 예측 값
		y_prediction_ra_plot=np.vstack([y_measured_ra_3h,ra_prediction_1h]) #합치기
		y_prediction_en_plot=np.vstack([y_measured_en_3h,ra_en_prediction_1h]) #합치기

		## 과거 3시간 + 향후 1시간 혼합 엔탈피 예측 값
		ma_en_p=np.zeros(length_ma_past)
		for i in range(length_ma_past):
			x[i]=(y_prediction_ra_plot[i]-y_prediction_ma_plot[i])/(y_prediction_ma_plot[i]-oa_t_past[i])
			ma_en_p[i]=(x[i]/(x[i]+1))*OA_en_past[i]+(1/(x[i]+1))*y_prediction_en_plot[i]
			
		#"예측 데이터 + 제어 혼합 엔탈피 계산"
		# 혼합온도 예측+제어 값[과거 3시간 예측+미래 1시간 제어]
		y_prediction_ma_c_1h=y_prediction_ma_c[12,:].reshape(4,1) #미래 1시간 제어 값
		y_prediction_ma_plot_c=np.vstack([y_prediction_ma_3h,y_prediction_ma_c_1h]) #합치기    
		ma_en_c=np.zeros(length_ma_past)
		for i in range(length_ma_past):
			x[i]=(y_prediction_ra_plot[i]-y_prediction_ma_plot_c[i])/(y_prediction_ma_plot_c[i]-oa_t_past[i])
			ma_en_c[i]=(x[i]/(x[i]+1))*OA_en_past[i]+(1/(x[i]+1))*y_prediction_en_plot[i]

			ma_en_past_bems[i]=(x[i]/(x[i]+1))*OA_en_past[i]+(1/(x[i]+1))*RA_en_past[i]
	# %% 사용자 입출력 변수 선정
	"1. 사용자 입력 변수 "
	"   1) 입력 변수#1:  AHU 선택 기능 -> ahu_name=str('AHU4 (지상2층)')" #공조기 이름
	"   2) 입력 변수#2: 현재 시간 기준 실시간 OFF/ON 기능 -> real_time = 0 #실시간 OFF/ON [0,1]"
	"   3) 입력 변수#3: 현재 시간 기준 실시간 OFF 일 경우, 사용자가 예측&제어 날짜 설정 -> past_year, past_month, past_today, past_hour, past_minute"
	"   4) 입력 변수#4: 공기조화기 별 전열교환기 설치 유무 설정 기능 -> 아이파크타워 ahu_heat_ex = 0 "
	"2. 출력 변수"
	"   1) 과거 3시간 BEMS 데이터 + 향후 1시간 제어 OA댐퍼 개도율 -> 최종output: ahu_damper_control_c_plot"
	"   2) 과거 3시간 예측 데이터 + 향후 1시간 제어 혼합온도 -> 최종output: y_prediction_ma_plot_c" 
	"   3) 과거 3시간 예측 데이터 + 향후 1시간 예측 혼합온도 -> 최종output: y_prediction_ma_plot" 
	"   4) 과거 3시간 예측 데이터 + 향후 1시간 제어 혼합엔탈피 -> 최종output: ahu_ma_en_c_plot_wh"
	"   5) 과거 3시간 예측 데이터 + 향후 1시간 예측 혼합 엔탈피 -> 최종output: ahu_ma_en_p_plot"
	# %% 혼합온도 및 댐퍼 개도율 제어 결과 출력
	#plt.rcParams["font.family"] = "Times New Roman"
	#plt.rcParams["font.size"] = 20
	#plt.rcParams["figure.figsize"] = (15,8) 


	"혼합온도 및 댐퍼 개도율"
	if real_time == 0:  
		# 댐퍼 개도율 측정 값
		ahu_damper_control_plot=ahu_damper_test_ma_past[3:19,:]
		# 댐퍼 개도율 제어 값
		ahu_damper_control_c_plot=np.vstack([ahu_damper_test_ma_past[3:15,:], ahu_damper_control.reshape(4,1)]) # 과거 3시간 댐퍼 개도율 + 향후 1시간 제어 댐퍼 개도율
	elif real_time == 1:  
		# 혼합온도 BEMS 값
		if len(ahu_ma_t_test_3h)>12:
			ahu_ma_t_test_3h=ahu_ma_t_test_3h[1:][:]
		ma_t_past=np.vstack([ahu_ma_t_test_3h,y_prediction_ma_p.reshape(4,1)])
		# 혼합온도 예측 값
		y_prediction_ma_plot=np.vstack([y_test_prediction_ma_3h_plot,y_prediction_ma_p.reshape(4,1)])
		# 혼합온도 제어 값
		y_prediction_ma_plot_c=np.vstack([y_test_prediction_ma_3h_plot, y_prediction_ma_c.reshape(4,1)])
		# 댐퍼 개도율 측정 값
		ahu_damper_control_bems_future= np.zeros([4,1])
		ahu_damper_control_bems_future[:,:] = ahu_damper_test_3h[11,:]
		ahu_damper_control_plot=np.vstack([ahu_damper_test_3h, ahu_damper_control_bems_future.reshape(4,1)])
		# 댐퍼 개도율 제어 값
		ahu_damper_control_c_plot=np.vstack([ahu_damper_test_3h, ahu_damper_control.reshape(4,1)]) # 과거 3시간 댐퍼 개도율 + 향후 1시간 제어 댐퍼 개도율

	"혼합 엔탈피 및 댐퍼 개도율"
	if real_time == 0:  
		# 단위 환산 J/kg -> wh/kg
		ahu_ma_en_test_plot_wh=ma_en_bems*0.00027
		ahu_ma_en_c_plot_wh= ma_en_c*0.00027
		ahu_ma_en_p_plot_wh=ma_en_p*0.00027
	elif real_time == 1:
		# 혼합엔탈피 BEMS 값
		ma_en_bems=np.vstack([ma_en_3h_bems.reshape(12,1),ma_en_1h_p.reshape(4,1)])
		# 혼합엔탈피 예측 값
		ma_en_p=np.vstack([ma_en_3h_p.reshape(12,1),ma_en_1h_p.reshape(4,1)])
		# 혼합온도 제어 값
		ma_en_c=np.vstack([ma_en_3h_p.reshape(12,1),ma_en_1h_c.reshape(4,1)])
		# 댐퍼 개도율 측정 값
		ahu_damper_control_bems_future= np.zeros([4,1])
		ahu_damper_control_bems_future[:,:] = ahu_damper_test_3h[11,:]
		ahu_damper_control_plot=np.vstack([ahu_damper_test_3h, ahu_damper_control_bems_future]) 
		 # 단위 환산 J/kg -> wh/kg
		ahu_ma_en_test_plot_wh=ma_en_bems*0.00027
		ahu_ma_en_c_plot_wh= ma_en_c*0.00027
		ahu_ma_en_p_plot_wh=ma_en_p*0.00027

	from realtime_xtick import fn_realtime_xtick
	if minute>44 and minute<60:
		hour=copy.deepcopy(hour)+1

	"혼합온도 및 댐퍼 개도율 그래프화"
	if real_time == 0:  
		# 댐퍼 개도율 측정 값
		ahu_damper_control_plot=ahu_damper_test_ma_past[3:19,:]
		# 댐퍼 개도율 제어 값
		ahu_damper_control_c_plot=np.vstack([ahu_damper_test_ma_past[3:15,:], ahu_damper_control.reshape(4,1)]) # 과거 3시간 댐퍼 개도율 + 향후 1시간 제어 댐퍼 개도율
	elif real_time == 1:  
		# 혼합온도 BEMS 값
		if len(ahu_ma_t_test_3h)>12:
			ahu_ma_t_test_3h=ahu_ma_t_test_3h[1:][:]
		ma_t_past=np.vstack([ahu_ma_t_test_3h,y_prediction_ma_p.reshape(4,1)])
		# 혼합온도 예측 값
		y_prediction_ma_plot=np.vstack([y_test_prediction_ma_3h_plot,y_prediction_ma_p.reshape(4,1)])
		# 혼합온도 제어 값
		y_prediction_ma_plot_c=np.vstack([y_test_prediction_ma_3h_plot, y_prediction_ma_c.reshape(4,1)])
		# 댐퍼 개도율 측정 값
		ahu_damper_control_bems_future= np.zeros([4,1])
		ahu_damper_control_bems_future[:,:] = ahu_damper_test_3h[11,:]
		ahu_damper_control_plot=np.vstack([ahu_damper_test_3h, ahu_damper_control_bems_future.reshape(4,1)])
		# 댐퍼 개도율 제어 값
		ahu_damper_control_c_plot=np.vstack([ahu_damper_test_3h, ahu_damper_control.reshape(4,1)]) # 과거 3시간 댐퍼 개도율 + 향후 1시간 제어 댐퍼 개도율

	Result = {'ahu_damper_BEMS': ahu_damper_control_plot[:12,0], 'ahu_damper_pred': ahu_damper_control_plot[12:,0],
			  'ahu_damper_control': ahu_damper_control_c_plot,
			  'MA temp_BEMS': ma_t_past,
			  'y_prediction_ma': y_prediction_ma_plot,
			  'y_prediction_ma_c': y_prediction_ma_plot_c,
			  'Enthalpy_BEMS': ahu_ma_en_test_plot_wh,
			  'y_prediction_Enthalpy': ahu_ma_en_p_plot_wh,
			  'y_prediction_c_1_Enthalpy': ahu_ma_en_c_plot_wh,
			  'label': fn_realtime_xtick(hour,minute),
			  'PredictedDay': str(year) + ' - ' + str(month) + ' - ' + str(today),
			  'msg': 'Success - Done'}

	return Result
	


'''
fig, axes = plt.subplots(nrows=4, ncols=1,figsize=(28, 15))
sty_label = {'fontweight': 'bold', 'fontsize': 19, 'labelpad': 14}
sty_tick = {'labelsize': 18, 'pad': 10}
sty_minortick = {'labelsize': 18, 'pad': 10}
length=16
ahu_damper_test_3h = ahu_damper_control_plot[:12,0]# 과거 3시간 댐퍼 개도율
ahu_damper_control_bems_future = ahu_damper_control_plot[12:,0]# 미래 1시간 댐퍼 개도율
# 외기 댐퍼 측정&예측 값
axes[0].plot(np.arange(0,12),ahu_damper_test_3h,'r',linestyle='-')# 과거 3시간 BEMS 댐퍼 개도율
axes[0].plot(np.arange(0,16),ahu_damper_control_plot,'c',linestyle='--')  # 미래 1시간 예측 댐퍼 개도율
axes[0].legend(['Damper opening_measured', 'Damper opening_predicted'])
axes[0].set_ylabel('OA damper opening[%]', **sty_label)
axes[0].set_xlim(0,15)
axes[0].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
axes[0].set_ylim(0,120)
axes[0].set_yticks([_*20 for _ in range(6)])

# OA 댐퍼 제어 값
axes[1].plot(np.arange(0,length),ahu_damper_control_c_plot,'k',linestyle='-') # 미래 1시간 제어 댐퍼 개도율
axes[1].grid(color='#BDBDBD', linestyle='-', linewidth=2, )
axes[1].set_ylabel('OA damper opening[%]', **sty_label) 
axes[1].legend(['Damper opening_controlled'])
axes[1].grid()
axes[1].set_xlim(0,15,1)
axes[1].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
axes[1].set_ylim(0,120)
axes[1].set_yticks([_*20 for _ in range(6)])

# 혼합 온도 측정, 예측, 제어값
axes[2].plot(np.arange(0,length),ma_t_past,'r',linestyle='-')  # 혼합온도 측정값
axes[2].plot(np.arange(0,length),y_prediction_ma_plot,'c',linestyle='-') # 혼합온도 예측값
axes[2].plot(np.arange(0,length),y_prediction_ma_plot_c,'b',linestyle='--')  # 혼합온도 제어값
axes[2].grid(color='#BDBDBD', linestyle='-', linewidth=2, )
axes[2].set_ylabel('MA temperature[℃]', **sty_label) 
axes[2].legend(['MA temp_measured','MA temp_predicted','MA temp_controlled'])
axes[2].grid()
axes[2].set_xlim(0,15,1)
axes[2].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
axes[2].set_ylim(0,30)
axes[2].set_yticks([_*6 for _ in range(6)])

# 혼합 엔탈피 측정, 예측, 제어값
axes[3].plot(np.arange(0,length),ahu_ma_en_test_plot_wh,'r',linestyle='-') # 혼합엔탈피 측정값
axes[3].plot(np.arange(0,length),ahu_ma_en_p_plot_wh,'c',linestyle='-') # 혼합엔탈피 예측값
axes[3].plot(np.arange(0,length),ahu_ma_en_c_plot_wh,'b',linestyle='--') # 혼합엔탈피 제어값
axes[3].grid(color='#BDBDBD', linestyle='-', linewidth=2, )
axes[3].set_ylabel('MA enthalpy[Wh/kg]', **sty_label)
axes[3].legend(['MA enthaply_measured', 'MA enthaply_predicted','MA enthalpy_controlled'])
axes[3].grid()
axes[3].set_xlim(0,15,1)
axes[3].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
axes[3].set_yticks([_*3 for _ in range(6)])

labels = []
label= fn_realtime_xtick(hour,minute)
xticklabel =label
axes[0].set_xticklabels(xticklabel)
axes[1].set_xticklabels(xticklabel)
axes[2].set_xticklabels(xticklabel)
axes[3].set_xticklabels(xticklabel)

axes[0].grid()
axes[1].grid()
axes[2].grid()
axes[3].grid()
plt.savefig(str(month) + str(today) + "_Ipark_" + ahu_name + '_Enthalpy_control_result.png')
print("댐퍼 개도율 100% 조건 만족 여부 ","enthalpy:",OA_en < y_prediction_en.reshape(4,1), "temp:", OA_t<y_prediction_ra.reshape(4,1),"OA_hr:", OA_hr<0.012)
print("댐퍼 개도율 30% 조건 만족 여부 ","enthalpy:",OA_en > y_prediction_en.reshape(4,1), "damper_opening:", ahu_damper_control>30)
plt.show()
'''