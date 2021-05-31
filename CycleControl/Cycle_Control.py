# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 20:46:55 2021

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
import datetime
import pickle
import os.path
from IPT_oncycle_model import HDC_oncycle_IPT, HDC_Predict_IPT, fn_checkerror
from HDC_XL import HDC_XL, HDC_IPARK_XL


import os

def main_CC(currentDateTime, InputParam_CC, CurrentAHUidx):
	os.chdir('D:/IPTower/CycleControl')  # move directory to current file
		
	Result = {'AHUStatus_measured': None, 'RA_measured': None, 'AHUStatus_controlled': None, 'RA_controlled': None, 'AHUStatus_optimul': None, 'RA_optimul': None, 'PredictedDay': None, 'msg': None}
	
	# %% 파라미터 설정
	x_train_batch_size = 100 # 배치 사이즈 결정
	epochs=500 # 모델 epoch 설정
	# ------------------------------------------------------
	# 사용자 입력 데이터
	Cycle_onoff = InputParam_CC[CurrentAHUidx]['CycleStatus'] # 사용자 입력 사이클
	Cycle_SP_cool = InputParam_CC[CurrentAHUidx]['UpperTemp'] #사용자 입력 쾌적 온도 상한
	Cycle_SP_heat = InputParam_CC[CurrentAHUidx]['LowerTemp'] #사용자 입력 쾌적 온도 하한
	Cycle_start = InputParam_CC[CurrentAHUidx]['InitialTime'] # 사이클 제어 시작 시간
	Cycle_end = InputParam_CC[CurrentAHUidx]['FinalTime'] # 사이클 제어 종료 시간
		
	Month_start = InputParam_CC[CurrentAHUidx]['InitialCoolMonth'] # 해당 월부터 냉방 시작
	Month_end = InputParam_CC[CurrentAHUidx]['FinalCoolMonth'] # 해당 다음 달부터 냉방 종료
	AHU_name = InputParam_CC[CurrentAHUidx]['AHU'] # 공조기 이름 입력하면 필요 데이터 추출
	#AHU_name = 'AHU6 (지상7층)' # 공조기 이름 입력하면 필요 데이터 추출
	
	# 현재 가능한 공조기 목록
	# AHU4 (지상2층)
	# AHU6 (지상3층)
	# AHU6 (지상4층)
	# AHU6 (지상5층)
	# AHU6 (지상6층)
	# AHU6 (지상7층)
	# AHU6 (지상8층)
	# AHU6 (지상9층)
	'''
	# %% 데이터 준비

	# ------------------------------------------------------
	# 사용자 입력 데이터
	Cycle_onoff = [1, 1, 1, 0] # 사용자 입력 사이클
	Cycle_SP_cool = 26 #사용자 입력 쾌적 온도 상한
	Cycle_SP_heat = 18 #사용자 입력 쾌적 온도 하한
	Cycle_start = 9 # 사이클 제어 시작 시간
	Cycle_end = 18 # 사이클 제어 종료 시간

	Month_start = 7 # 해당 월부터 냉방 시작
	Month_end = 10 # 해당 월부터 냉방 종료


	# ------------------------------------------------------

	# 현재 날짜 및 시각 load
	now = datetime.datetime.now()
	year_ = int(now.strftime('%Y'))
	month_ = int(now.strftime('%m'))
	day_ = int(now.strftime('%d'))
	hour_ = int(now.strftime('%H'))
	minute_ = int(now.strftime('%M'))

	# -------------------------------------------------------
	# 실시간 테스트용 임의입력
	year_ = 2020
	month_ = 11
	day_ = 16
	hour_ = 14
	'''

	# ------------------------------------------------------
	# 현재 날짜 및 시각 load
	# now = datetime.datetime.now()
	#now = currentDateTime
	#year_ = int(now.strftime('%Y'))
	#month_ = int(now.strftime('%m'))
	#day_ = int(now.strftime('%d'))
	#hour_ = int(now.strftime('%H'))
	#minute_ = int(now.strftime('%M'))
	# 실시간 테스트용 임의입력
	year_ = 2020
	month_ = 11
	day_ = 16
	hour_ = 14
	# -------------------------------------------------------
	# 기계학습 모델 저장 및 로드
	tag_save_weights = '.h5' # 파일 확장자
	tag_save_scalor_x = '_x.gz'
	tag_save_scalor_y = '_y.gz'
	tag_RA = '_RA'
	tag_SA = '_SA'
	tag_OA = '_OA'

	# 모델 저장 변수 이름
	model_name_RA = AHU_name + tag_RA + tag_save_weights
	model_name_SA = AHU_name + tag_SA + tag_save_weights 
	model_name_OA = AHU_name + tag_OA + tag_save_weights 

	# 모델 저장 변수 이름
	scalor_name_x_RA = AHU_name + tag_RA + tag_save_scalor_x
	scalor_name_y_RA = AHU_name + tag_RA + tag_save_scalor_y
	scalor_name_x_SA = AHU_name + tag_SA + tag_save_scalor_x
	scalor_name_y_SA = AHU_name + tag_SA + tag_save_scalor_y
	scalor_name_x_OA = AHU_name + tag_OA + tag_save_scalor_x
	scalor_name_y_OA = AHU_name + tag_OA + tag_save_scalor_y

	path_model_RA = './models_Cycle/' + model_name_RA
	path_model_SA = './models_Cycle/' + model_name_SA
	path_model_OA = './models_Cycle/' + model_name_OA

	# %% 파라미터 설정
	x_train_batch_size = 100 # 배치 사이즈 결정
	epochs=500 # 모델 epoch 설정

	# 난방 및 냉방기간 결정(내부변수)
	if month_ >= Month_start and month_ <= Month_end:
		Season = '냉방'
	else:
		Season = '난방' 

	# %% 모델 업데이트 및 예측

	# 모델 업데이트를 위한 시간 및 기존 모델 존재여부 체크
	save_name = './models_Cycle/' + model_name_RA + '_check_time.p'
	
	if os.path.exists(save_name): # 가동시간 체크용 파일이 존재하는지 확인
		# 존재할 경우 : 비교대상 로드 및 체크 후 업데이트 여부 결정
		
		with open(save_name,'rb') as check_time:
			check_day = pickle.load(check_time) # 비교대상 날짜 로드
		
		if (check_day == day_) and os.path.exists(path_model_RA): # 모듈 가동 날짜 및 모델 존재여부 체크
			# 저장된 날짜와 모듈 가동 날짜가 동일한 경우 & 해당 공조기의 모델이 존재하는 경우 : 모델 업데이트 X, 모델 로드
			# print('1')
			pass
			
					
		else:   
			# 저장된 날짜와 모듈 가동 날짜가 동일하지 않은 경우 또는 모델이 존재하지 않는 경우 : 모델 업데이트 O, 날짜 저장, 모델 저장 
			# print('2')
			HDC_oncycle_IPT(AHU_name, model_name_RA, model_name_SA, model_name_OA
						, scalor_name_x_RA, scalor_name_y_RA, scalor_name_x_SA
						, scalor_name_y_SA, scalor_name_x_OA, scalor_name_y_OA
						, year_, month_, day_)
			
			
	else:
		# 존재하지 않을 경우 : 모델 생성, 날짜 저장, 모델 저장   
		# print('3')
		HDC_oncycle_IPT(AHU_name, model_name_RA, model_name_SA, model_name_OA
						, scalor_name_x_RA, scalor_name_y_RA, scalor_name_x_SA
						, scalor_name_y_SA, scalor_name_x_OA, scalor_name_y_OA
						, year_, month_, day_)


	RA_test_save, RA_pred_save, RA_cont_save, RA_opt_save, Cycle_pred, Cycle_onoff, Cycle_opt_b = HDC_Predict_IPT(model_name_RA
				 ,model_name_SA,model_name_OA
				 ,scalor_name_x_RA, scalor_name_x_SA, scalor_name_x_OA
				 ,AHU_name, year_, month_, day_, hour_, Cycle_onoff
				 ,Cycle_start, Cycle_end, Cycle_SP_cool, Cycle_SP_heat
				 ,Season)

	if not RA_test_save.size == 1: # RA_test_save를 테스트하여 빈값이 아니면 모듈 계속 수행(정상일 경우 4 출력됨, 빈값은 운영시간이 아닐 경우 추력됨)
		# %% 모델 오류 검출 및 재연산
		check_test = fn_checkerror(RA_test_save,model_name_RA,Season,RA_pred_save
						  ,RA_cont_save,RA_opt_save)
		
		if check_test == 1: # 모델이 오류있다고 판단될 시 모델 재생성 및 예측
			n_try = 0 # 반복 횟수
			while not (check_test == 0 or n_try == 9): # 최대 10회 반복
				HDC_oncycle_IPT(AHU_name, model_name_RA, model_name_SA, model_name_OA
						, scalor_name_x_RA, scalor_name_y_RA, scalor_name_x_SA
						, scalor_name_y_SA, scalor_name_x_OA, scalor_name_y_OA
						, year_, month_, day_)
				RA_test_save, RA_pred_save, RA_cont_save, RA_opt_save, Cycle_pred, Cycle_onoff, Cycle_opt_b = HDC_Predict_IPT(model_name_RA
					 ,model_name_SA,model_name_OA
					 ,scalor_name_x_RA, scalor_name_x_SA, scalor_name_x_OA
					 ,AHU_name, year_, month_, day_, hour_, Cycle_onoff
					 ,Cycle_start, Cycle_end, Cycle_SP_cool, Cycle_SP_heat
					 ,Season)
		
				check_test = fn_checkerror(RA_test_save,model_name_RA,Season,RA_pred_save
										   ,RA_cont_save,RA_opt_save)
				n_try = n_try + 1
		
		if (check_test == 1 and n_try == 9):
			# 반복했음에도 불구하고 적절한 모델을 개발하지 못한 경우 기존운전대로
			print('기계학습 연산 어려움. 기존 운영방침에 따라 운전하십시오')
			RA_cont_save = copy.copy(RA_pred_save)
			RA_opt_save = copy.copy(RA_pred_save)
			Cycle_onoff = copy.copy(Cycle_pred)
			Cycle_opt_b = copy.copy(Cycle_pred) 
			os.remove(path_model_RA) #저장 모델 삭제(해당 파일 삭제 시 모델 재생성 가능)
		
		
		# %% 이전 3시간 저장파일 불러오기
			
		
		Plot_RA_pred = np.zeros((1, 12)) # 예측 데이터 저장 위치    
		Plot_RA_pred_t1 = np.zeros((1, 4)) # 예측 데이터 저장 위치    
		Plot_RA_pred_t2 = np.zeros((1, 4)) # 예측 데이터 저장 위치    
		Plot_RA_pred_t3 = np.zeros((1, 4)) # 예측 데이터 저장 위치    
		Plot_Cycle_pred = np.zeros((1, 12)) # 예측 데이터 저장 위치
		   
		
		hour_t1 = hour_ - 3 # 3 시간전 데이터
		hour_t2 = hour_ - 2 # 2 시간전 데이터
		hour_t3 = hour_ - 1 # 1 시간전 데이터
		
	# %% 3시간전 데이터    
		if (hour_t1 >= Cycle_start and hour_t1 <= Cycle_end):
			
			# 과거 운영데이터를 받아 기동상태 출력
			db = HDC_IPARK_XL()
			AHU_on_db = db.retrieve_from_db(
				[
												
					(AHU_name,'급기팬 운전상태'), 
				],
				(year_, month_, day_, hour_t1, 0), 4,  # 1시간 과거 데이터
			)
			AHU_on = AHU_on_db.values
			Cycle_mea = copy.copy(AHU_on)
			
			RA_test_save_t1, RA_pred_save_t1, RA_cont_save_t1, RA_opt_save_t1, Cycle_pred_t1, Cycle_onoff_t1, Cycle_opt_b_t1 = HDC_Predict_IPT(model_name_RA
				 ,model_name_SA,model_name_OA
				 ,scalor_name_x_RA, scalor_name_x_SA, scalor_name_x_OA
				 ,AHU_name, year_, month_, day_, hour_t1, Cycle_mea
				 ,Cycle_start, Cycle_end, Cycle_SP_cool, Cycle_SP_heat
				 ,Season)
			# 모듈에 Cycle_mea를 이용 control 값을 조정 및 도출
			
			Cycle_pred_t1 = copy.copy(Cycle_mea.reshape(1,-1))
			
		else:
		# %% 1~3시간전이 설정시간을 벗어날 경우 측정데이터와 동일하게 출력
			db = HDC_IPARK_XL()
			RA_alter_db = db.retrieve_from_db(
				[
					
					(AHU_name,'AHU 환기(RA)온도'),
					(AHU_name,'급기팬 운전상태'),
					
				],
				(year_, month_, day_, hour_t1, 0), 4,  # 1시간 과거 데이터
			)
			
			
			RA_alter = RA_alter_db.values
			RA_pred_save_t1 = copy.copy(RA_alter[:,0].reshape(1,-1))
			RA_cont_save_t1 = copy.copy(RA_alter[:,0].reshape(1,-1))
			Cycle_pred_t1 = copy.copy(RA_alter[:,1].reshape(1,-1))
			
			
	# %% 2시간전 데이터        
		if (hour_t2 >= Cycle_start and hour_t2 <= Cycle_end):
			
			# 과거 운영데이터를 받아 기동상태 출력
			db = HDC_IPARK_XL()
			AHU_on_db = db.retrieve_from_db(
				[
					
					(AHU_name,'급기팬 운전상태'), 
					
				],
				(year_, month_, day_, hour_t2, 0), 4,  # 2시간 과거 데이터
			)
				   
			AHU_on = AHU_on_db.values
			Cycle_mea = copy.copy(AHU_on)
			
			RA_test_save_t2, RA_pred_save_t2, RA_cont_save_t2, RA_opt_save_t2, Cycle_pred_t2, Cycle_onoff_t2, Cycle_opt_b_t2 = HDC_Predict_IPT(model_name_RA
				 ,model_name_SA,model_name_OA
				 ,scalor_name_x_RA, scalor_name_x_SA, scalor_name_x_OA
				 ,AHU_name, year_, month_, day_, hour_t2, Cycle_mea
				 ,Cycle_start, Cycle_end, Cycle_SP_cool, Cycle_SP_heat
				 ,Season)
			
			Cycle_pred_t2 = copy.copy(Cycle_mea.reshape(1,-1))
		
		else:
		# %% 1~3시간전이 설정시간을 벗어날 경우 측정데이터와 동일하게 출력
			db = HDC_IPARK_XL()
			RA_alter_db = db.retrieve_from_db(
				[
					
					(AHU_name,'AHU 환기(RA)온도'),
					(AHU_name,'급기팬 운전상태'),
					
				],
				(year_, month_, day_, hour_t2, 0), 4,  # 2시간 과거 데이터
			)
			
					
			RA_alter = RA_alter_db.values
			RA_pred_save_t2 = copy.copy(RA_alter[:,0].reshape(1,-1))
			RA_cont_save_t2 = copy.copy(RA_alter[:,0].reshape(1,-1))
			Cycle_pred_t2 = copy.copy(RA_alter[:,1].reshape(1,-1))
			
			
	# %% 1시간전 데이터        
		if (hour_t3 >= Cycle_start and hour_t3 <= Cycle_end):
			
			# 과거 운영데이터를 받아 기동상태 출력
			db = HDC_IPARK_XL()
			AHU_on_db = db.retrieve_from_db(
				[
					
					(AHU_name,'급기팬 운전상태'), 
					
				],
				(year_, month_, day_, hour_t3, 0), 4,  # 1시간 과거 데이터
			)
					
			AHU_on = AHU_on_db.values
			Cycle_mea = copy.copy(AHU_on)
			
			RA_test_save_t3, RA_pred_save_t3, RA_cont_save_t3, RA_opt_save_t3, Cycle_pred_t3, Cycle_onoff_t3, Cycle_opt_b_t3 = HDC_Predict_IPT(model_name_RA
				 ,model_name_SA,model_name_OA
				 ,scalor_name_x_RA, scalor_name_x_SA, scalor_name_x_OA
				 ,AHU_name, year_, month_, day_, hour_t3, Cycle_mea
				 ,Cycle_start, Cycle_end, Cycle_SP_cool, Cycle_SP_heat
				 ,Season)
			
			Cycle_pred_t3 = copy.copy(Cycle_mea.reshape(1,-1))
		
		else:
		# %% 1~3시간전이 설정시간을 벗어날 경우 측정데이터와 동일하게 출력
			db = HDC_IPARK_XL()
			RA_alter_db = db.retrieve_from_db(
				[
					
					(AHU_name,'AHU 환기(RA)온도'),
					(AHU_name,'급기팬 운전상태'),
					
				],
				(year_, month_, day_, hour_t3, 0), 4,  # 1시간 과거 데이터
			)
			
			
			RA_alter = RA_alter_db.values
			RA_pred_save_t3 = copy.copy(RA_alter[:,0].reshape(1,-1))
			RA_cont_save_t3 = copy.copy(RA_alter[:,0].reshape(1,-1))
			Cycle_pred_t3 = copy.copy(RA_alter[:,1].reshape(1,-1))
	   
		Cycle_pred_t1 = np.array(Cycle_pred_t1).reshape(1,-1)
		Cycle_pred_t2 = np.array(Cycle_pred_t2).reshape(1,-1)
		Cycle_pred_t3 = np.array(Cycle_pred_t3).reshape(1,-1)
		Cycle_mea_t1 = np.array(Cycle_pred_t1).reshape(1,-1)
		Cycle_mea_t2 = np.array(Cycle_pred_t2).reshape(1,-1)
		Cycle_mea_t3 = np.array(Cycle_pred_t3).reshape(1,-1)
		
		Plot_RA_pred = np.concatenate([RA_cont_save_t1,RA_cont_save_t2,RA_cont_save_t3],axis=1)
		
		Plot_RA_mea = copy.copy(Plot_RA_pred)
		Plot_Cycle_mea = np.concatenate([Cycle_mea_t1,Cycle_mea_t2,Cycle_mea_t3],axis=1)
		
		# %% UI 출력 데이터
		db = HDC_IPARK_XL()
		measured_db = db.retrieve_from_db(
			[
				
				(AHU_name,'AHU 환기(RA)온도'),
				(AHU_name,'급기팬 운전상태'),
				
			],
			(year_, month_, day_, hour_, 0), -12,  # 3시간 과거 데이터
		)
		
		AHU_measured = measured_db.values
		
		
		# # ------------------------------------------------------
		print(AHU_measured[:,0]) # 측정된 RA 데이터 (테스트용)
		# print(AHU_measured[:,1]) # 측정된 on off 데이터 (테스트용)
		
		print(RA_pred_save) # 모델 예측 RA 데이터
		print(RA_cont_save) # 제어 로직 RA 데이터
		print(RA_opt_save)  # 최적 제어 RA 데이터
		# print(RA_test_save)  # 최적 제어 RA 데이터
		
		print(Cycle_pred) # 기존 제어 공조기 가동 상태 
		print(Cycle_onoff) # 사이클 제어 공조기 가동 상태 
		print(Cycle_opt_b)  # 최적 제어 공조기 가동 상태 
		
		print(Plot_RA_mea) # 과거 3시간 데이터
		print(Plot_Cycle_mea) # 과거 3시간 운영데이터
		
	# # ------------------------------------------------------
	
		if str(type(Cycle_opt_b)) == "<class 'numpy.ndarray'>":
			Cycle_opt_b_ = Cycle_opt_b[0].tolist()
		else:
			Cycle_opt_b_ = Cycle_opt_b
		PredictedDay = str(year_) + ' - ' + str(month_) + ' - ' + str(day_)	
		Result = {'AHUStatus_measured': AHU_measured[:,1], 'RA_measured': AHU_measured[:,0], 'AHUStatus_controlled': Plot_Cycle_mea[0].tolist() + Cycle_onoff, 'RA_controlled': Plot_RA_mea[0].tolist() + RA_cont_save[0].tolist(), 'AHUStatus_optimul': Plot_Cycle_mea[0].tolist() + Cycle_opt_b_, 'RA_optimul': Plot_RA_mea[0].tolist() + RA_opt_save[0].tolist(), 'PredictedDay': PredictedDay, 'msg': 'Success - Done'}
	else:
		Result = {'AHUStatus_measured': None, 'RA_measured': None, 'AHUStatus_controlled': None, 'RA_controlled': None, 'AHUStatus_optimul': None, 'RA_optimul': None, 'PredictedDay': PredictedDay, 'msg': 'Failure - 현재 사용자 입력 기준 업무 시간이 아닙니다.'}
	return Result
	
'''	
# %% As-is To-be 비교 그래프
plt.figure(figsize=(12, 6))
plt.rcParams["font.size"] = 16
# x=np.arange(hour_+0.25,hour_+1.25,0.25).reshape(-1,1)

# plt.plot(Plot_RA_pred.reshape(-1,1),color='red')
# plt.plot(Plot_RA_cont.reshape(-1,1),'--', color='blue')
# plt.plot(Plot_RA_opt.reshape(-1,1),'--', color='green')
# plt.plot(AHU_measured[:,0].reshape(-1,1),'--', color='black')
# plt.plot(Plot_RA_mea.reshape(-1,1),color='red')

plt.plot(AHU_measured[:,0].reshape(-1,1),'--', color='blue')
plt.plot(Plot_RA_mea.reshape(-1,1),'--', color='green')

plt.xlabel('Time-steps(hour)')
plt.legend(['Predict','Control','optimal','measured'], loc='upper right')
plt.grid(True)
plt.tight_layout()    

db = HDC_IPARK_XL()
measured_db = db.retrieve_from_db(
	[
		
		(AHU_name,'AHU 환기(RA)온도'),
		
	],
	(year_, month_, day_, hour_, 0), 4,  # 1시간 측정 데이터
)

AHU_measured = measured_db.values
		
plt.figure(figsize=(12, 6))
plt.rcParams["font.size"] = 16
# x=np.arange(hour_+0.25,hour_+1.25,0.25).reshape(-1,1)

plt.plot(AHU_measured.reshape(-1,1),color='black')
plt.plot(RA_pred_save.reshape(-1,1),color='red')
plt.plot(RA_cont_save.reshape(-1,1),'--', color='blue')
plt.plot(RA_opt_save.reshape(-1,1),'--', color='green')


plt.xlabel('Time-steps(hour)')
plt.legend(['Predict','Control','optimal'], loc='upper right')
plt.grid(True)
plt.tight_layout()  
'''




