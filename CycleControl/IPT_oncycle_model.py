# -*- coding: utf-8 -*-
"""


@author: bslabCHK
"""

# %% import packages
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import copy
import sys;sys.path.append('..')
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import joblib 
# from HDC_XL import HDC_DB
import datetime
import pickle
import os.path
from HDC_XL import HDC_XL, HDC_IPARK_XL



# %% 
def HDC_oncycle_IPT(AHU_name, model_name_RA, model_name_SA, model_name_OA
				, scalor_name_x_RA, scalor_name_y_RA, scalor_name_x_SA
				, scalor_name_y_SA, scalor_name_x_OA, scalor_name_y_OA
				, year_, month_, day_):
	
	
	# %% 파라미터 설정
	x_train_batch_size = 100 # 배치 사이즈 결정
	epochs=500 # 모델 epoch 설정
	
	AHU_name = AHU_name # 공조기 이름 입력하면 필요 데이터 추출
	
		
	# 현재 날짜 및 시각 load
	# now = datetime.datetime.now()
	year_ = year_
	month_ = month_
	day_ = day_
		   
	# 모델 저장 변수 이름
	model_name_RA = model_name_RA
	model_name_SA = model_name_SA 
	model_name_OA = model_name_OA 
	
	# 모델 저장 변수 이름
	scalor_name_x_RA = scalor_name_x_RA
	scalor_name_y_RA = scalor_name_y_RA
	scalor_name_x_SA = scalor_name_x_SA
	scalor_name_y_SA = scalor_name_y_SA
	scalor_name_x_OA = scalor_name_x_OA
	scalor_name_y_OA = scalor_name_y_OA
	path_sc_RA = './models_Cycle/' + scalor_name_x_RA
	path_sc_SA = './models_Cycle/' + scalor_name_x_SA
	path_sc_OA = './models_Cycle/' + scalor_name_x_OA
	
	# 저장용 폴더 없을 경우 생성
	if not os.path.exists('./models_Cycle'):
		os.makedirs('./models_Cycle')
	 
		
	# %% 모델 훈련 또는 로드
	
	# -------------------------------------------------------
	# Case = 2 # 테스트용(모델생성 강제 스킵 등)
	
	# -------------------------------------------------------
	
	
		
	# SQL 데이터 불러오기
	db = HDC_IPARK_XL()
	sub_db_1_1 = db.retrieve_from_db(
		[
			
			(AHU_name,'AHU 환기(RA)온도'),
			('외부','외기온도'),
			(AHU_name,'AHU 급기(SA)온도'),            
			(AHU_name,'급기팬 운전상태'), 
		],
		(year_, month_, day_, 0, 0), -1344,  # 2주치 데이터
	)
	
	AHU_train = sub_db_1_1.values # 모델 훈련 x 데이터 저장
	
	
	# %% 모델 훈련용 데이터 정렬
		
	RA_x = [] # 환기건구온도 모델 입력 데이터 정렬
	RA_y = AHU_train[5:,0] # 환기건구온도 모델 출력 데이터 정렬
	OA_x = [] # 외기건구온도 모델 입력 데이터 정렬
	OA_y = AHU_train[5:,1] # 외기건구온도 모델 출력 데이터 정렬
	SA_x = [] # 급기건구온도 모델 입력 데이터 정렬
	SA_y = AHU_train[5:,2] # 급기건구온도 모델 출력 데이터 정렬
	
	# AHUOn_x = [] # 공조기 기동상태 입력 데이터 정렬
	# AHUOn_y = AHU_train[4:,3] # 공조기 기동상태 모델 출력 데이터 정렬
	# CCI_x = [] # 냉수코일입구 모델 입력 데이터 정렬
	# CCI_y = [] # 냉수코일입구 모델 출력 데이터 정렬
	   
	
	# t-4 ~ t 시간에 대하여 데이터 정렬
	for uu in range(0, int(len(AHU_train))-5):
		RA_x.append(np.concatenate((  
			AHU_train[uu:uu+4,0], # RA
			AHU_train[uu:uu+4,1], # OA
			AHU_train[uu:uu+4,2], # SA
			AHU_train[uu:uu+4,3], # AHU on off(과거)
			[AHU_train[uu+4,3]] # AHU on off(미래)
			)))
		SA_x.append(np.concatenate((
			# AHU_train[uu:uu+4,0], # CCI 
			AHU_train[uu:uu+4,0], # RA 
			AHU_train[uu:uu+4,1], # OA
			AHU_train[uu:uu+4,2], # SA
			AHU_train[uu:uu+4,3], # AHU on off(과거)
			[AHU_train[uu+4,3]] # AHU on off(미래)
			)))
		OA_x.append(
			AHU_train[uu:uu+4,1]
			)
		
		
	RA_x = np.array(RA_x)  
	SA_x = np.array(SA_x)
	OA_x = np.array(OA_x)
	
	sc_x_RA, sc_y_RA = MinMaxScaler(), MinMaxScaler()
	sc_x_SA, sc_y_SA = MinMaxScaler(), MinMaxScaler()
	sc_x_OA, sc_y_OA = MinMaxScaler(), MinMaxScaler()
	
	sc_x_RA.fit(RA_x)
	sc_x_SA.fit(SA_x)
	sc_x_OA.fit(OA_x)
	
	joblib.dump(sc_x_RA, path_sc_RA)
	joblib.dump(sc_x_SA, path_sc_SA)
	joblib.dump(sc_x_OA, path_sc_OA)  
	
	x_tr_RA = sc_x_RA.transform(RA_x)
	
	x_tr_SA = sc_x_SA.transform(SA_x)
	
	x_tr_OA = sc_x_OA.transform(OA_x)
	   
	# %% ANN 모델 구축 및 저장

	model_RA=keras.models.Sequential([keras.Input(shape=(17,)), 
									keras.layers.Dense(50, activation='relu'),
									keras.layers.Dense(50, activation='relu'),
									keras.layers.Dense(50, activation='relu'),
									keras.layers.Dense(1)])
	model_RA.compile(loss='mse', optimizer='adam', metrics="MeanAbsoluteError")
	model_RA.fit(x_tr_RA, RA_y, x_train_batch_size, epochs)
	
	model_SA=keras.models.Sequential([keras.Input(shape=(17,)), 
									keras.layers.Dense(50, activation='relu'),
									keras.layers.Dense(50, activation='relu'),
									keras.layers.Dense(50, activation='relu'),
									keras.layers.Dense(1)])
	model_SA.compile(loss='mse', optimizer='adam', metrics="MeanAbsoluteError")
	model_SA.fit(x_tr_SA, SA_y, x_train_batch_size, epochs)
	
	model_OA=keras.models.Sequential([keras.Input(shape=(4,)), 
									keras.layers.Dense(50, activation='relu'),
									keras.layers.Dense(50, activation='relu'),
									keras.layers.Dense(50, activation='relu'),
									keras.layers.Dense(1)])
	model_OA.compile(loss='mse', optimizer='adam', metrics="MeanAbsoluteError")
	model_OA.fit(x_tr_OA, OA_y, x_train_batch_size, epochs)
	
	# %% 모델 및 날짜 저장
	
	path_model_RA = './models_Cycle/' + model_name_RA
	path_model_SA = './models_Cycle/' + model_name_SA
	path_model_OA = './models_Cycle/' + model_name_OA
	model_RA.save(path_model_RA)
	model_SA.save(path_model_SA)
	model_OA.save(path_model_OA)
	print("Saved model to disk") 
	
	check_day = copy.copy(day_) # 저장용 변수에 현재 날짜 저장
	save_name = './models_Cycle/' + model_name_RA + '_check_time.p'
	with open(save_name,'wb') as check_time:
		pickle.dump(check_day, check_time) # 비교대상 날짜 저장
	
	# %% 훈련 데이터 정확도
	
	# y_train=model_RA.predict(x_tr_RA) # 훈련 기간 출력 변수 예측
	
	# Result_tr = y_train # 훈련결과 1열 저장
	# Result_Y = RA_y # 훈련기간 측정데이터
	
	# y_train_RMSE = np.sqrt(np.sum((Result_tr.T-Result_Y.T)**2)/len(Result_tr.T))
	# y_train_CVRMSE = y_train_RMSE/np.mean(Result_tr)*100
	# y_train_MBE = np.sum(Result_tr.T-Result_Y.T)/sum(Result_tr)*100
		
def HDC_Predict_IPT(model_name_RA,model_name_SA,model_name_OA
			 ,scalor_name_x_RA, scalor_name_x_SA, scalor_name_x_OA
			 ,AHU_name, year_, month_, day_, hour_, Cycle_onoff
			 ,Cycle_start, Cycle_end, Cycle_SP_cool, Cycle_SP_heat
			 ,Season):
	
	os.chdir('D:/IPTower/CycleControl')  # modified by HDC 
	path_model_RA = './models_Cycle/' + model_name_RA
	path_model_SA = './models_Cycle/' + model_name_SA
	path_model_OA = './models_Cycle/' + model_name_OA
	path_sc_RA = './models_Cycle/' + scalor_name_x_RA
	path_sc_SA = './models_Cycle/' + scalor_name_x_SA
	path_sc_OA = './models_Cycle/' + scalor_name_x_OA
	
	# 모델 로드
	model_RA = keras.models.load_model(path_model_RA)
	model_SA = keras.models.load_model(path_model_SA)
	model_OA = keras.models.load_model(path_model_OA)
		
	# 데이터 정규화 정보 로드
	
	sc_x_RA = joblib.load(path_sc_RA) 
	sc_x_SA = joblib.load(path_sc_SA)
	sc_x_OA = joblib.load(path_sc_OA)
	  
	print("Loaded model from disk")
		
	# return model_RA, model_SA, model_OA
		
		
	# %% 실시간 예측    
# def HDC_Predict(): 
	# SQL 데이터 불러오기
	db = HDC_IPARK_XL()
	sub_db_2 = db.retrieve_from_db(
		[
			
			(AHU_name,'AHU 환기(RA)온도'),
			('외부','외기온도'),
			(AHU_name,'AHU 급기(SA)온도'),            
			(AHU_name,'급기팬 운전상태'), 
		],
		(year_, month_, day_, hour_, 0), -4,  # 1시간 데이터
	)
	
	AHU_predcit = sub_db_2.values #실시간 예측을 위한 데이터 저장
	# AHU_measured = test_data.values # 비교용 데이터(테스트용)
	   
	
	# %% 실시간 예측용 데이터 정렬    
	
	RA_p_x = [] # 환기건구온도 모델 예측 데이터 정렬
	OA_p_x = [] # 외기건구온도 모델 예측 데이터 정렬
	SA_p_x = [] # 급기건구온도 모델 예측 데이터 정렬
	
	RA_p_x.append(np.concatenate((  
				AHU_predcit[0:4,0], # RA
				AHU_predcit[0:4,1], # OA
				AHU_predcit[0:4,2], # SA
				AHU_predcit[0:4,3], # AHU on off(과거)
				[1] # AHU on off(미래)
				)))
	SA_p_x.append(np.concatenate((
				# AHU_predcit[0:4,0], # CCI
				AHU_predcit[0:4,0], # RA 
				AHU_predcit[0:4,1], # OA
				AHU_predcit[0:4,2], # SA
				AHU_predcit[0:4,3], # AHU on off(과거)
				[1] # AHU on off(미래)
				)))
	OA_p_x.append(
				AHU_predcit[0:4,1]
				)
			
	RA_pred = np.array(RA_p_x)  
	SA_pred = np.array(SA_p_x)
	OA_pred = np.array(OA_p_x)
	
	
	
	# %% 모델 예측 및 제어
	# 사용자 입력 사이클 제어 입력 변수
	RA_cont = copy.copy(RA_pred) 
	SA_cont = copy.copy(SA_pred)
	OA_cont = copy.copy(OA_pred)
	
	RA_opt = copy.copy(RA_pred) 
	SA_opt = copy.copy(SA_pred)
	OA_opt = copy.copy(OA_pred)
	
	RA_test = copy.copy(RA_pred) 
	SA_test = copy.copy(SA_pred)
	OA_test = copy.copy(OA_pred)
	
	
	RA_pred_t = copy.copy(RA_pred)
	RA_pred_t1 = np.zeros((1, np.shape(RA_cont)[1]))
	RA_pred_save = np.zeros((1, 4))
	
	SA_pred_t = copy.copy(SA_pred)
	SA_pred_t1 = np.zeros((1, np.shape(SA_cont)[1]))
	SA_pred_save = np.zeros((1, 4))
	
	OA_pred_t = copy.copy(OA_pred)
	OA_pred_t1 = np.zeros((1, np.shape(OA_cont)[1]))
	OA_pred_save = np.zeros((1, 4))
	
	RA_cont_t = copy.copy(RA_cont)
	RA_cont_t1 = np.zeros((1, np.shape(RA_cont)[1]))
	RA_cont_save = np.zeros((1, 4))
	
	SA_cont_t = copy.copy(SA_cont)
	SA_cont_t1 = np.zeros((1, np.shape(SA_cont)[1]))
	SA_cont_save = np.zeros((1, 4))
	
	OA_cont_t = copy.copy(OA_cont)
	OA_cont_t1 = np.zeros((1, np.shape(OA_cont)[1]))
	OA_cont_save = np.zeros((1, 4))
	
	RA_opt_t = copy.copy(RA_opt)
	RA_opt_t1 = np.zeros((1, np.shape(RA_opt)[1]))
	RA_opt_save = np.zeros((1, 4))
	RA_opt_save_t = np.zeros((1, 4))
	
	SA_opt_t = copy.copy(SA_opt)
	SA_opt_t1 = np.zeros((1, np.shape(SA_opt)[1]))
	SA_opt_save = np.zeros((1, 4))
	SA_opt_save_t = np.zeros((1, 4))
	
	OA_opt_t = copy.copy(OA_opt)
	OA_opt_t1 = np.zeros((1, np.shape(OA_opt)[1]))
	OA_opt_save = np.zeros((1, 4))
	OA_opt_save_t = np.zeros((1, 4))
	
	RA_test_t = copy.copy(RA_pred)
	RA_test_t1 = np.zeros((1, np.shape(RA_cont)[1]))
	RA_test_save = np.zeros((1, 4))
	
	SA_test_t = copy.copy(SA_pred)
	SA_test_t1 = np.zeros((1, np.shape(SA_cont)[1]))
	SA_test_save = np.zeros((1, 4))
	
	OA_test_t = copy.copy(OA_pred)
	OA_test_t1 = np.zeros((1, np.shape(OA_cont)[1]))
	OA_test_save = np.zeros((1, 4))
	
	
	
	Cycle_pred = [1,1,1,1]
	# Cycle_onoff = [1,1,0,0] # 사이클 테스트용 
	Cycle_opt = np.ones((1,len(Cycle_onoff)))
	Cycle_test = [0,0,0,0]
	
	
	
	
			
	if (hour_ >= Cycle_start and hour_ <= Cycle_end):
		# %% 기존 로직(1시간 연속 가동)
		
		for ii in range(0,4):        
			
			RA_pred_t[0,16] = Cycle_pred[ii] # 제어변수 입력
			SA_pred_t[0,16] = Cycle_pred[ii] # 제어변수 입력
			
			# print(RA_pred_t)
			
			RA_pred_sc = sc_x_RA.transform(RA_pred_t)
			RA_pred_y = model_RA.predict(RA_pred_sc) # RA 출력
							
			SA_pred_sc = sc_x_SA.transform(SA_pred_t)
			SA_pred_y = model_SA.predict(SA_pred_sc) # SA 출력
					
			OA_pred_sc = sc_x_OA.transform(OA_pred_t)
			OA_pred_y = model_OA.predict(OA_pred_sc) # OA 출력
			
			# 다음 timestep을 위한 변수 준비
			RA_pred_t1[0,:np.shape(RA_pred)[1]-1] = RA_pred_t[0,1:]
			RA_pred_t1[0,3] = RA_pred_y
			RA_pred_t1[0,7] = OA_pred_y
			RA_pred_t1[0,11] = SA_pred_y
			# RA_pred_t1[0,15] = 1 # 제어변수
			# RA_pred_t1[0,16] = 1 # 제어변수
			RA_pred_t = copy.copy(RA_pred_t1)
			
			# print(RA_pred_t1)
			
			SA_pred_t1[0,:np.shape(SA_pred)[1]-1] = SA_pred_t[0,1:]
			SA_pred_t1[0,3] = RA_pred_y
			SA_pred_t1[0,7] = OA_pred_y
			SA_pred_t1[0,11] = SA_pred_y
			# SA_pred_t1[0,15] = 1 # 제어변수
			# SA_pred_t1[0,16] = 1 # 제어변수
			SA_pred_t = copy.copy(SA_pred_t1)
					
			OA_pred_t1[0,:3] = OA_pred_t[0,1:]
			OA_pred_t1[0,3] = OA_pred_y  
			OA_pred_t = copy.copy(OA_pred_t1) 
			
			#예측 결과
			RA_pred_save[0,ii] = copy.copy(RA_pred_y)
			SA_pred_save[0,ii] = copy.copy(SA_pred_y)
			OA_pred_save[0,ii] = copy.copy(OA_pred_y)
			
		# %% 사용자 설정 사이클 제어        
			
		for ii in range(0,4):
					
			RA_cont_t[0,16] = Cycle_onoff[ii] # 제어변수 입력
			SA_cont_t[0,16] = Cycle_onoff[ii] # 제어변수 입력        
			
			# print(RA_cont_t)
			
			RA_cont_sc = sc_x_RA.transform(RA_cont_t)
			RA_cont_y = model_RA.predict(RA_cont_sc) # RA 출력
							
			SA_cont_sc = sc_x_SA.transform(SA_cont_t)
			SA_cont_y = model_SA.predict(SA_cont_sc) # SA 출력
					
			OA_cont_sc = sc_x_OA.transform(OA_cont_t)
			OA_cont_y = model_OA.predict(OA_cont_sc) # OA 출력
		
			# 다음 timestep을 위한 변수 준비
			RA_cont_t1[0,:np.shape(RA_cont)[1]-1] = RA_cont_t[0,1:]
			RA_cont_t1[0,3] = RA_cont_y
			RA_cont_t1[0,7] = OA_cont_y
			RA_cont_t1[0,11] = SA_cont_y
			# RA_cont_t1[0,15] = Cycle_onoff[ii] # 제어변수
			# RA_cont_t1[0,12] = Cycle_onoff[ii] # 제어변수 입력
			RA_cont_t = copy.copy(RA_cont_t1)
			
			# print(RA_cont_t1)
			
			SA_cont_t1[0,:np.shape(SA_cont)[1]-1] = SA_cont_t[0,1:]
			SA_cont_t1[0,3] = RA_cont_y
			SA_cont_t1[0,7] = OA_cont_y
			SA_cont_t1[0,11] = SA_cont_y
			# SA_cont_t1[0,15] = Cycle_onoff[ii] # 제어변수
			# SA_cont_t1[0,16] = Cycle_onoff[ii] # 제어변수 입력
			SA_cont_t = copy.copy(SA_cont_t1)
			
					
			OA_cont_t1[0,:3] = OA_cont_t[0,1:]
			OA_cont_t1[0,3] = OA_cont_y  
			OA_cont_t = copy.copy(OA_cont_t1) 
			
			#예측 결과
			RA_cont_save[0,ii] = copy.copy(RA_cont_y)
			SA_cont_save[0,ii] = copy.copy(SA_cont_y)
			OA_cont_save[0,ii] = copy.copy(OA_cont_y)
	
		# 쾌적온도 벗어날 경우 검출
		check_cont_c = np.array(np.where(RA_cont_save[0,:]>Cycle_SP_cool)) 
		check_cont_h = np.array(np.where(RA_cont_save[0,:]<Cycle_SP_heat))     
			
		if check_cont_c.size or check_cont_h.size:
			print('쾌적온도 벗어남') # 사용자 설정 사이클 제어 시 쾌적온도범위를 벗어날 경우 경고 출력
			
		# %% 최적 사이클 제어
		Cycle_opt = np.ones((1,len(Cycle_onoff)))
		for uu in range(0, len(Cycle_onoff)):
			Cycle_opt[0,len(Cycle_onoff)-uu-1] = 0
			
			# RA_opt_save_t = copy.copy(RA_opt_y)
			# SA_opt_save_t = copy.copy(SA_opt_y)
			# OA_opt_save_t = copy.copy(OA_opt_y)
			RA_opt_t = copy.copy(RA_opt)
			SA_opt_t = copy.copy(SA_opt)
			OA_opt_t = copy.copy(OA_opt)
			for kk in range(0, len(Cycle_onoff)):
				RA_opt_t[0,16] = Cycle_opt[0,kk] # 제어변수 입력
				SA_opt_t[0,16] = Cycle_opt[0,kk] # 제어변수 입력        
				
				RA_opt_sc = sc_x_RA.transform(RA_opt_t)
				RA_opt_y = model_RA.predict(RA_opt_sc) # RA 출력
								
				SA_opt_sc = sc_x_SA.transform(SA_opt_t)
				SA_opt_y = model_SA.predict(SA_opt_sc) # SA 출력
						
				OA_opt_sc = sc_x_OA.transform(OA_opt_t)
				OA_opt_y = model_OA.predict(OA_opt_sc) # OA 출력
			
				# 다음 timestep을 위한 변수 준비
				RA_opt_t1[0,:np.shape(RA_opt)[1]-1] = RA_opt_t[0,1:]
				RA_opt_t1[0,3] = RA_opt_y
				RA_opt_t1[0,7] = OA_opt_y
				RA_opt_t1[0,11] = SA_opt_y
				RA_opt_t = copy.copy(RA_opt_t1)
				
				# print(RA_opt_t1)
				
				SA_opt_t1[0,:np.shape(SA_opt)[1]-1] = SA_opt_t[0,1:]
				SA_opt_t1[0,3] = RA_opt_y
				SA_opt_t1[0,7] = OA_opt_y
				SA_opt_t1[0,11] = SA_opt_y
				SA_opt_t = copy.copy(SA_opt_t1)
				
						
				OA_opt_t1[0,:3] = OA_opt_t[0,1:]
				OA_opt_t1[0,3] = OA_opt_y  
				OA_opt_t = copy.copy(OA_opt_t1) 
				
				#예측 결과
				RA_opt_save_t[0,kk] = copy.copy(RA_opt_y)
				SA_opt_save_t[0,kk] = copy.copy(SA_opt_y)
				OA_opt_save_t[0,kk] = copy.copy(OA_opt_y)
	
			# print(Cycle_opt)
			# print(RA_opt_save_t)
			
				
			# 쾌적온도 벗어날 경우 검출
			check_opt_c = np.array(np.where(RA_opt_save_t[0,:]>Cycle_SP_cool)) 
			check_opt_h = np.array(np.where(RA_opt_save_t[0,:]<Cycle_SP_heat))    
			
			# print(check_opt_c)
			# print(check_opt_h)
				
			if check_opt_c.size or check_opt_h.size:
				# print('최적 시나리오 중 쾌적온도 벗어나는 case 존재') # 최적 사이클 제어 방안 중 쾌적온도범위를 벗어날 경우 경고 출력 
				pass
			else:
				# 최적 제어 시도 성공 시 결과 저장
				RA_opt_save = copy.copy(RA_opt_save_t)
				SA_opt_save = copy.copy(SA_opt_save_t)
				OA_opt_save = copy.copy(OA_opt_save_t)
				Cycle_opt_b = copy.copy(Cycle_opt)
			
			if not 'Cycle_opt_b' in locals():
				# 최적 제어 도출 실패 시 기존 운영과 동일하게 설정 
				RA_opt_save = copy.copy(RA_pred_save)
				SA_opt_save = copy.copy(SA_pred_save)
				OA_opt_save = copy.copy(OA_pred_save)
				Cycle_opt_b = copy.copy(Cycle_pred)
				
				# 데이터 확인용
				# RA_opt_save = copy.copy(RA_opt_save_t)
				# SA_opt_save = copy.copy(SA_opt_save_t)
				# OA_opt_save = copy.copy(OA_opt_save_t)
				# Cycle_opt_b = copy.copy(Cycle_opt)
				
		# %% 모델 오류 검출 (냉방/난방기간 예상 거동과 다를경우 오류 경고)
		# print(Cycle_opt_b)
		for ii in range(0,4):        
			
			RA_test_t[0,16] = Cycle_test[ii] # 제어변수 입력
			SA_test_t[0,16] = Cycle_test[ii] # 제어변수 입력
			
			RA_test_sc = sc_x_RA.transform(RA_test_t)
			RA_test_y = model_RA.predict(RA_test_sc) # RA 출력
							
			SA_test_sc = sc_x_SA.transform(SA_test_t)
			SA_test_y = model_SA.predict(SA_test_sc) # SA 출력
					
			OA_test_sc = sc_x_OA.transform(OA_test_t)
			OA_test_y = model_OA.predict(OA_test_sc) # OA 출력
			
			# 다음 timestep을 위한 변수 준비
			RA_test_t1[0,:np.shape(RA_test)[1]-1] = RA_test_t[0,1:]
			RA_test_t1[0,3] = RA_test_y
			RA_test_t1[0,7] = OA_test_y
			RA_test_t1[0,11] = SA_test_y
			RA_test_t = copy.copy(RA_test_t1)
			# print(RA_test_t1)
			
			SA_test_t1[0,:np.shape(SA_test)[1]-1] = SA_test_t[0,1:]
			SA_test_t1[0,3] = RA_test_y
			SA_test_t1[0,7] = OA_test_y
			SA_test_t1[0,11] = SA_test_y
			SA_test_t = copy.copy(SA_test_t1)
					
			OA_test_t1[0,:3] = OA_test_t[0,1:]
			OA_test_t1[0,3] = OA_test_y  
			OA_test_t = copy.copy(OA_test_t1) 
			
			#예측 결과
			RA_test_save[0,ii] = copy.copy(RA_test_y)
			SA_test_save[0,ii] = copy.copy(SA_test_y)
			OA_test_save[0,ii] = copy.copy(OA_test_y)          
			
		
			
		
	
				   
		
		
		
	else:
		print('사용자 입력 시간 벗어남') # 사용자가 설정한 제어 시작 및 끝시간 이외의 시간에 모듈 가동하였을 시 예측하지 않음  
		RA_test_save = np.array([0]) # 계산 수행되지 않을 시 빈값 내보냄
		RA_pred_save = np.array([0])
		RA_cont_save = np.array([0])
		RA_opt_save = np.array([0])
		Cycle_pred = np.array([0])
		Cycle_onoff = np.array([0])
		Cycle_opt_b = np.array([0])
				  
	
	return RA_test_save, RA_pred_save, RA_cont_save, RA_opt_save, Cycle_pred, Cycle_onoff, Cycle_opt_b
		
	
# %% 오류 체크    
def fn_checkerror(RA_test_save,model_name_RA,Season,RA_pred_save
				  ,RA_cont_save,RA_opt_save):
	cof_safe = 0.5 # 안전 범위 계수    
	check_test = 0
	if Season == '냉방':
		for jj in range(0,3):
			if (RA_test_save[0,jj] - RA_test_save[0,jj+1]) > cof_safe:
				check_test = 1
		 
	elif Season == '난방':
		for jj in range(0,3):
			if (RA_test_save[0,jj+1] - RA_test_save[0,jj]) > cof_safe:
				check_test = 1
	
	if check_test:
		print('모델 오류 예상됨') # AHU 모델 이상으로 판정하여 다시 모델 생성하도록 저장파일 삭제
		# os.remove(model_name_RA) #저장 모델 삭제(해당 파일 삭제 시 모델 재생성 가능)
		check_test = 1
	# %% 예측된 온도가 정상범주에서 벗어날 경우 경고(모델 재생성 필요)
	limit_ub = 50 # 온도 상한선
	limit_lb = 10 # 온도 하한선
	check_pred_f_ub = np.array(np.where(RA_pred_save[0,:]>=limit_ub))
	check_pred_f_lb = np.array(np.where(RA_pred_save[0,:]<=limit_lb))
	check_cont_f_ub = np.array(np.where(RA_cont_save[0,:]>=limit_ub))
	check_cont_f_lb = np.array(np.where(RA_cont_save[0,:]<=limit_lb))
	check_opt_f_ub = np.array(np.where(RA_opt_save[0,:]>=limit_ub))
	check_opt_f_lb = np.array(np.where(RA_opt_save[0,:]<=limit_lb))
	
	if check_pred_f_ub.size or check_pred_f_lb.size:
		print('기본제어 예측치가 실내온도 범주를 벗어남')
		# os.remove(model_name_RA) #저장 모델 삭제(해당 파일 삭제 시 모델 재생성 가능)
		check_test = 1
	if check_cont_f_ub.size or check_cont_f_lb.size:
		print('사용자 제어 예측치가 실내온도 범주를 벗어남')
		# os.remove(model_name_RA) #저장 모델 삭제(해당 파일 삭제 시 모델 재생성 가능)
		check_test = 1
	if check_opt_f_ub.size or check_opt_f_lb.size:
		print('최적제어 예측치가 실내온도 범주를 벗어남')
		# os.remove(model_name_RA) #저장 모델 삭제(해당 파일 삭제 시 모델 재생성 가능)
		check_test = 1
	return check_test
	