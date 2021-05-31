# -*- coding: utf-8 -*-

from HDC_AHU_Model import HDC_IPARK_AHU
from HDC_XL import HDC_IPARK_XL
import os

def main_AOC(currentDateTime, InputParam_AOC, AHUidx_AOC):
	os.chdir(r'D:\IPTower\AHUOptimulControl\\')  # move directory to current file
	
	Result = {'AHU_Start': None, 'AHU_Stop': None, 'PredictedDate': None, 'msg': None}
	xl = HDC_IPARK_XL()

	#ahu_3f = HDC_IPARK_AHU('지상3층', now=(2020, 11, 29), db=xl)
	ahu_3f = HDC_IPARK_AHU(InputParam_AOC[AHUidx_AOC]['AHU'][6:10], now=(2020, 11, 29), db=xl)
	if InputParam_AOC[AHUidx_AOC]['SSRadioBtn'] == 'Start':
		res_start = ahu_3f.optimal_control(
			start_time  = (2020, 12, 3, 5, 15),  # 'now',
			is_start    = True,
			target      = (2020, 12, 3, 9),
			target_temp = 25,
			is_cool     = False,
			sch         = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
			margin      = 0.15,
			update      = True,
		)
		Result['AHU_Start'] = res_start
		Result['PredictedDate'] = res_start['time']
		
		# Output 정리
		print(res_start['summary'])      # 결과 요약
		print(res_start['time'])         # 기동/정지 시작 시점
		print(res_start['success'])      # 성공 여부
		print(res_start['table'])        # 입력된 공조기, 냉동기 기동 상태와 RA 온도 (예측 이전, 이후)
		print(res_start['target temp'])  # 목표 온도
		print(res_start['df'])           # 공조기 기동/정지 시간에 따른 제어 결과들
		
	else:
		# 냉방 최적 정지 제어
		res_stop = ahu_3f.optimal_control(
			start_time  = (2020, 12, 3, 18),  # 'now',
			is_start    = False,
			target      = (2020, 12, 3, 20),
			target_temp = 23,
			is_cool     = False,
			sch         = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
			margin      = 0.15,
			update      = True,
		)
		Result['AHU_Stop'] = res_stop
		Result['PredictedDate'] = res_stop['time']
		
		print(res_stop['summary'])       # 결과 요약
		print(res_stop['time'])          # 기동/정지 시작 시점
		print(res_stop['success'])       # 성공 여부
		print(res_stop['table'])         # 입력된 공조기, 냉동기 기동 상태와 RA 온도 (예측 이전, 이후)
		print(res_stop['target temp'])   # 목표 온도
		print(res_stop['df'])            # 공조기 기동/정지 시간에 따른 제어 결과들

	Result['msg'] = 'Success - Done'
	
	return Result