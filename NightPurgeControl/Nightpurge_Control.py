import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import datetime
from matplotlib import gridspec
from RealTimeDataAccumulator import get_weather_forecast
import psychrolib as psy
import os

def main_NC(currentDateTime, InputParam_NC, AHUidx_NC):
	os.chdir(r'D:\IPTower\NightPurgeControl\\')

	#================================< SQL 연결 >================================#
	from HDC_XL import HDC_IPARK_XL
	db = HDC_IPARK_XL('DB/')  # 엑셀, csv 파일의 위치 지정

	Result = {'title': None, 'future_num': None, 'x_label': None, 'RA_asis': None,
					  'RA_tobe': None, 'RA_fail': None,
					  'OA': None, 'RA_proposal': None,
					  'damp_on_best': None, 'Damper_fail': None, 'damp_asis': None,
					  'damper_ratio': None, 'AHU_on_proposal': None, 'load_saving': None,
					  'msg': None}

	#======================< 나이트퍼지 제어 검증 시작 시간 >======================#

	#   - 실시간인 경우
	now_time = datetime.datetime.now()
	now_year = now_time.year
	now_month = now_time.month
	now_day = now_time.day
	now_hour = now_time.hour
	now_minute = (now_time.minute // 15) * 15
	UI_now = (now_year, now_month, now_day, now_hour, now_minute)

	#   - 사용자 검증기간 입력할 경우
	UI_year = 2020
	UI_month = 11
	UI_day = 1
	UI_hour = 20
	UI_minute = 0
	UI_time = (UI_year, UI_month, UI_day, UI_hour, UI_minute)

	#   - 실시간 연결 : UI_now
	#   - 희망 검증 날짜 연결 : UI_time (단, 직접 입력해야함)
	now = UI_time



	#===============================< 사용자 입력 >===============================#

	# 입력 1. 냉방기간 설정 (시간 ~ 끝 month)
	#initial_month = 5       # 예 : [4, 5, 6, 7]
	#final_month = 11        # 예 : [9, 10, 11]
	initial_month = InputParam_NC[AHUidx_NC]['InitialCoolMonth']  # 예 : [4, 5, 6, 7]
	final_month = InputParam_NC[AHUidx_NC]['FinalCoolMonth']  # 예 : [9, 10, 11]

	# 입력 2. 나이트 퍼지 희망 공조기 (string)
	# 'AHU4 (지상2층)', 'AHU6 (지상3층)', 'AHU6 (지상4층)', 'AHU6 (지상5층),
	# 'AHU6 (지상6층)', 'AHU6 (지상7층)', 'AHU6 (지상8층)', 'AHU6 (지상9층)'

	#UI_AHU = 'AHU4 (지상2층)'
	UI_AHU = str(InputParam_NC[AHUidx_NC]['AHU'])

	# 입력 3. 나이트 종료 시작 (시, 분) : 종료 시간 ( 가능한 선택시간 : 21시~6시)
	#end_hour = 6
	#end_minute = 0
	end_hour = InputParam_NC[AHUidx_NC]['end_hour']
	end_minute = InputParam_NC[AHUidx_NC]['end_minute']

	# 입력 4. RA-OA (float) : 댐퍼를 개도시점 결정을 위한 최소 실내온도와 외기온도 차이값
	#temp_diff_RAOA = 3
	temp_diff_RAOA = InputParam_NC[AHUidx_NC]['delta_RAOA']

	# 입력 5. RA_target - 다음 제어 시퀀스로 인한 RA : 최적 댐퍼 개도 시퀀스 결정을 위한 허용 오차 (float)
	#temp_tolorance = 0.3
	temp_tolorance = InputParam_NC[AHUidx_NC]['margin_error']



	# <<<<<<<<< 사용자 입력 오류처리 >>>>>>>>> #

	# 오류 처리할 변수
	check_year = now[0]
	check_month = now[1]
	check_day = now[2]
	check_start_hour = now[3]
	check_start_minute = now[4]
	check_end_hour = end_hour
	check_end_minute = end_minute


	# 오류 1. 입력 month가 설정한 냉방기간에 속하지 않는 경우
	if check_month < initial_month or check_month > final_month :
		Result['msg'] = 'Failure - 지금은 냉방기간이 아닙니다.' 
		return Result

	# 오류 2. 입력 start_time이 나이트퍼지 시작시간에 속하지 않는 경우 
	if 6 < check_start_hour < 19:
		Result['msg'] = 'Failure - 지금은 나이트퍼지 가능한 시간이 아닙니다.'
		return Result

	# 오류 3. 입력 end_hour가 시작시간보다 빠른경우
	if (check_end_hour > 6 or (check_end_hour==6 and check_end_minute>0)) and check_end_hour < 21:
		Result['msg'] = 'Failure - 종료 시간은 21시 ~ 6시여야 합니다.'
		return Result

	# 오류 4. 입력 end_time이 나이트퍼지 가능한 시간에서 벗어난 경우    
	now_dt = datetime.datetime(check_year, check_month, check_day, check_start_hour, check_start_minute)
	end_dt = datetime.datetime(check_year, check_month, check_day, check_end_hour, check_end_minute)

	if 0 <= check_start_hour < 6:
		end_dt = now_dt + datetime.timedelta(hours=check_end_hour-check_start_hour)
		
	elif check_end_hour < 12:
		end_dt += datetime.timedelta(days=1)
		
	elif end_dt <= now_dt:
		Result['msg'] = 'Failure - 종료 시간은 시작 시간 이후여야 합니다.'
		return Result


	end_time = (end_dt.year, end_dt.month, end_dt.day, end_dt.hour, end_dt.minute)



	#===============================< 사용자 출력 >===============================#
	# 출력 1. 실내온도
	RA_asis = []    # 제어 안할 시
	RA_TOBE = []    # 제어 수행 시

	# 출력 2. 외기온도
	OA = []

	# 출력 3. 댐퍼 개도율
	damp_asis = []
	damp_on_best = []

	# 출력 4. 팬동력 사용량 & 부하 절감량
	fan_power = []
	load_saving = []



	#=============================< 데이터 불러오기 >=============================#
	# 변수 1. 현재시간 이전 RA (제어 알고리즘에 사용)
	RA_past = db.get_sequence((UI_AHU, 'AHU 환기(RA)온도'), now, -1)
	RH_past = db.get_sequence((UI_AHU, 'AHU 환기(RA)습도'), now, -1)

	# 변수 2. 현재시간 이전 OA (기상청 데이터 활용시 선형보간을 위해 사용)
	OA_past = db.get_sequence(('외부', '외기온도'), now, -1)
	OH_past = db.get_sequence(('외부', '외기습도'), now, -1)

	# 변수 3. 측정된 OA (모델 검증기간 입력시에 활용)
	OA_measured = db.get_sequence(('외부', '외기온도'), now, end_time)
	OH_measured = db.get_sequence(('외부', '외기습도'), now, end_time)



	#============================< 기상정보(OA) 연결 >============================#

	# 기상API 연결 : 향후 15분 외기온도 계산 (대구 정보)
	nx = str(89) 			# 예보지점 x 좌표
	ny = str(91) 			# 예보지점 y 좌표

	temp_wf, humid_wf = get_weather_forecast(nx, ny) # HDC측 기상 API 로드


	# 현재 기준 가장 가까운 미래 시간 불러오기
	fu_year = int(temp_wf[1][2][0:4])
	fu_month = int(temp_wf[1][2][4:6])
	fu_day = int(temp_wf[1][2][6:8])
	fu_hour = int(int(temp_wf[1][3])/100)
	fu_minute = 0

	today_hour_temp_p = int(temp_wf[1][4])           # 미래 시간: 외기온도
	today_hour_humid_p = int(humid_wf[1][4])         # 미래 시간: 외기습도


	# 현재 시간과 가장 가까운 미래 기상 데이터 시간 차이 (15분 단위)
	time_delta = datetime.datetime(fu_year, fu_month, fu_day, fu_hour, fu_minute) \
		- datetime.datetime(now[0], now[1], now[2], now[3], now[4])

	first_lin = int(time_delta.seconds/60/15)  # 현재 시간과 가장 가까운 미래 기상데이터 시간까지 선형보간 개수
		

	# 외기 건구온도 선형보간
	OA_future1 = np.linspace( OA_past[0],  int(temp_wf[1][4]), first_lin)
	OA_future2 = np.linspace( int(temp_wf[1][4]),  int(temp_wf[2][4]), 12)
	OA_future3 = np.linspace( int(temp_wf[2][4]),  int(temp_wf[3][4]), 12)
	OA_future4 = np.linspace( int(temp_wf[3][4]),  int(temp_wf[4][4]), 12)

	OA_future = np.hstack([OA_future1, OA_future2, OA_future3, OA_future4])


	# 외기 상대습도 선형보간
	OH_future1 = np.linspace( OH_past[0],  int(humid_wf[1][4]), first_lin)
	OH_future2 = np.linspace( int(humid_wf[1][4]),  int(humid_wf[2][4]), 12)
	OH_future3 = np.linspace( int(humid_wf[2][4]),  int(humid_wf[3][4]), 12)
	OH_future4 = np.linspace( int(humid_wf[3][4]),  int(humid_wf[4][4]), 12)

	OH_future = np.hstack([OH_future1, OH_future2, OH_future3, OH_future4])



	if now == UI_now:       # 실시간 연결시 --> 기상청 정보
		OA = OA_future
		OH = OH_future
		
	 
	else:                  # 사용자 희망 기간 입력시 --> BEMS 외기건구온도
		OA = OA_measured
		OH = OH_measured



	#============================< 엔탈피 계산 >============================#
	psy.SetUnitSystem(psy.SI)


	# OA 엔탈피
	OA_en = []

	for i in range(len(OA)):
		# 1. 절대습도 산출
		oa_hr = psy.GetHumRatioFromRelHum(OA[i], OH[i]/100, 101325)
		# 2. 엔탈피 산출
		oa_enth = psy.GetMoistAirEnthalpy(OA[i], oa_hr)
		# 3. 엔탈피 저장
		OA_en.append(oa_enth)


	# RA 엔탈피
	RA_hr = psy.GetHumRatioFromRelHum(RA_past, RH_past/100, 101325)
	RA_en = psy.GetMoistAirEnthalpy(RA_past, RA_hr)



	#===============================< 공조기 정보 >===============================#
	def info_AHU(UI_AHU):
		'''공조기 별 스펙 정보를 반환해주는 함수 '''

	   
		if UI_AHU == 'AHU4 (지상2층)':
			V = 17421    # m3
			Hz = 55
			OA_Q = 355*60      # CMH
			
		if UI_AHU == 'AHU6 (지상3층)':
			V = 17421    # m3
			Hz = 55
			OA_Q = 578*60      # CMH
		
		if UI_AHU == 'AHU6 (지상4층)':
			V = 17421    # m3
			Hz = 55
			OA_Q = 578*60      # CMH
		
		if UI_AHU == 'AHU6 (지상5층)':
			V = 17421    # m3
			Hz = 55
			OA_Q = 578*60      # CMH
			
		if UI_AHU == 'AHU6 (지상6층)':
			V = 17421    # m3
			Hz = 55
			OA_Q = 578*60      # CMH
			
		if UI_AHU == 'AHU6 (지상7층)':
			V = 17421    # m3
			Hz = 55
			OA_Q = 578*60      # CMH
			
		if UI_AHU == 'AHU6 (지상8층)':
			V = 17421    # m3
			Hz = 55
			OA_Q = 578*60      # CMH
			
		if UI_AHU == 'AHU6 (지상9층)':
			V = 17421    # m3
			Hz = 55
			OA_Q = 578*60      # CMH

		return V, Hz, OA_Q
			

	# V = 152*60/total_specQ*total_area
	V, Hz, OA_Q = info_AHU(UI_AHU)



	#=========================< RA(t) 모델 (1법칙 기반) >=========================#

	def RA_t(Volume, AHU_specQ, AHU_hz, damp_ratio, OA, RA_p):
		''' 현재시간 RA를 반환하는 함수'''
		rho_ = 1.2  # kg/m3 
		Cp = 1006  # J/(kgK)
		SA_Q = AHU_specQ * (AHU_hz/60) * (damp_ratio/100)*1/4
		bottom = rho_*Cp*Volume + rho_*Cp*SA_Q
		
		RA_new = (rho_*Cp*Volume*RA_p + (rho_*Cp*SA_Q)*OA) / bottom
			
		return RA_new



	#================================< 제어 시작 >================================#

	# <<<<< Pre-simulation (RA_asis) : 제어 안할 경우 RA >>>>> #

	# 기본값 및 제어값 
	damp_ratio = [0, 100]
	AHU_hz = [0, Hz]

	# 예보 데이터 수 (Number of iteration by Day)
	future_num = OA.shape[0]

	# 초기값 (댐퍼 -> 폐쇄)
	RA_asis = []
	RA_asis.append(RA_past[0])

	damp_off = []
	damp_off.append(damp_ratio[0])

	# Iteration by Day
	for fstep in range(1, future_num):
		
		# 현재 외기 온도 (기상예보)
		OA_now = OA[fstep]

		# 공조기 off 상태에서 RA 예측 (제어 reference 온도) -> asis
		RA_free2 = RA_t(V, OA_Q, AHU_hz[0], damp_ratio[0], OA_now, RA_asis[-1])
		RA_asis.append(RA_free2)
		damp_asis.append(damp_ratio[0])
		


	# <<<<< 최적 제어 시퀀스 탐색 (RA_tobe) : 제어 후 >>>>> #

	# 1. 제어 가능한 시점 탐색 (RA보다 OA가 temp_diff_RAOA도 이상인 시간)
	#   - 제어 가능한 시점이 없을 경우, 오류 메시지 출력
	on_condition_idx1 = np.array(np.where(temp_diff_RAOA < np.array(RA_asis)-OA)[0])    # 조건1 : OA<RA
	on_condition_idx2 = np.array(np.where(OA_en<RA_en)[0])                              # 조건2 : OA엔탈피<RA엔탈피

	on_condition_idx = [x for x in on_condition_idx1 if x in on_condition_idx2]


	if min(OA_en) >= RA_en:
		raise ValueError('나이트퍼지 제어 가능한 시간이 없습니다.')    

	if len(on_condition_idx) == 0:
		raise ValueError('나이트퍼지 제어 가능한 시간이 없습니다.')


	# 2. 제어 시퀀스 탐색 순서 정하기
	OA_lowest_idx = np.where(OA==min(OA))[0][0]     # OA 최저기온 시점 찾기
	lowest_end_length = (len(OA)-1)-OA_lowest_idx   # OA 최저기온 시점과 종료 시점 간 타임스텝

	# damp on 순서 만들기
	on_idx = []
	for i in range(lowest_end_length+1):
		on_idx.append(OA_lowest_idx+i)

	rest_idx = [x for x in on_condition_idx if x not in on_idx]

	for i in range(1, len(rest_idx)+1):
		on_idx.append(rest_idx[-i])
		
		
	# 3. 공조기운전 시퀀스 초기화
	AHU_on = np.zeros_like(OA)


	# 4. 결과 저장 장치 - 제어시퀀스 탐색 히스토리 
	RA_proposal = []
	AHU_on_proposal = []


	# 5. 종단온도 & 종단시퀀스 & 공조기 기동 횟수 기록 (업데이트)
	RA_best = 100
	AHU_on_best = 100


	# 6. RA 타깃 온도 -> 외기 최저 온도
	RA_target = np.random.random(1)+min(OA)

		
	# 7.  최적 제어 시퀀스 탐색
	for i in range(len(on_condition_idx)):  # Loop1. 제어 시퀀스 탐색 

		RA_tobe= []
		RA_tobe.append(RA_past[0])
		
		find_idx = on_idx[i]
		AHU_on[find_idx] = 1

			  
		for fstep in range(0, future_num):   # Loop2. 일일 탐색 

			# 현재 외기 온도 (기상예보)
			OA_now = OA[fstep]

			AHU_cond = int(AHU_on[fstep])
			RA_free = RA_t(V, OA_Q, AHU_hz[AHU_cond], damp_ratio[AHU_cond], OA_now, RA_tobe[-1])
			RA_tobe.append(RA_free)
			

		### -> Post processing  (타깃온도 조건에 맞으면 Stop)
		if  np.abs(RA_target - RA_tobe[-1]) <= temp_tolorance:  
			RA_best = RA_tobe[1:]
			AHU_on_best = AHU_on
			break
		
			
		# 현재 제어 기록 저장 (검토용)    
		RA_proposal.append(RA_tobe[1:])
		AHU_on_proposal.append(deepcopy(AHU_on))
	   
		
	if RA_best == 100:
		RA_best = RA_proposal[-1]
		AHU_on_best = AHU_on_proposal[-1]
		

	#===============================< 팬동력 계산 >===============================#
	def info_fan(UI_AHU):
		'''공조기 별 팬동력 정보를 반환해주는 함수 '''
		
		if UI_AHU == 'AHU4 (지상2층)':
			Sf_power = 15
			Rf_power = 5.5
			Sf_specQ = 355*60
			Rf_specQ = 355*60
			
		if UI_AHU == 'AHU6 (지상3층)':
			Sf_power = 18.5
			Rf_power = 11
			Sf_specQ = 578*60
			Rf_specQ = 553*60
		
		if UI_AHU == 'AHU6 (지상4층)':
			Sf_power = 18.5
			Rf_power = 11
			Sf_specQ = 578*60
			Rf_specQ = 553*60
			
		if UI_AHU == 'AHU6 (지상5층)':
			Sf_power = 18.5
			Rf_power = 11
			Sf_specQ = 578*60
			Rf_specQ = 553*60
			
		if UI_AHU == 'AHU6 (지상6층)':
			Sf_power = 18.5
			Rf_power = 11
			Sf_specQ = 578*60
			Rf_specQ = 553*60
			
		if UI_AHU == 'AHU6 (지상7층)':
			Sf_power = 18.5
			Rf_power = 11
			Sf_specQ = 578*60
			Rf_specQ = 553*60
			
		if UI_AHU == 'AHU6 (지상8층)':    
			Sf_power = 18.5
			Rf_power = 11
			Sf_specQ = 578*60
			Rf_specQ = 553*60
			
		if UI_AHU == 'AHU6 (지상9층)':
			Sf_power = 18.5
			Rf_power = 11
			Sf_specQ = 578*60
			Rf_specQ = 553*60
			
		return Sf_power, Rf_power, Sf_specQ, Rf_specQ


	Sf_power, Rf_power, Sf_specQ, Rf_specQ = info_fan(UI_AHU)



	#================================< 제어 결과 >================================#
	# 총 시도한 시퀀스 개수 (댐퍼개도율 리스트 개수)
	len_trial = len(AHU_on_proposal)


	# 출력 1. 실내온도
	RA_TOBE = RA_best


	# 출력 3. 댐퍼 개도율 (AHU_on -> Damper scale로 변환)
	damp_asis = [(AHU_on_proposal[i] * (damp_ratio[1] - damp_ratio[0]) + damp_ratio[0]) for i in range(len(AHU_on_proposal))]

	damp_TOBE = AHU_on_best
	damp_on_best = damp_TOBE * (damp_ratio[1] - damp_ratio[0]) + damp_ratio[0]


	# 출력 4. 팬동력 사용량 & 부하 절감량
	#   - 절감 열량 계산 파라미터
	rho_ = 1.2                                              # 단위: kg/m3 
	Cp = 1006                                               # 단위: J/(kgK)
	delta_RAtemp = RA_asis[-1] - RA_TOBE[-1]                # 단위: C
	on_time = 15*damp_TOBE.sum()/60


	# 부하 절감량
	load_saving_v = rho_*Cp*V*delta_RAtemp/1000/3600         # 단위: kWh
	load_saving = '%0.1f' %load_saving_v


	# 팬동력 사용량 계산 (단위: kWh)
	fan_power_v = (OA_Q/Sf_specQ*Sf_power + \
				  (OA_Q*(Rf_specQ/Sf_specQ)/Rf_specQ)*Rf_power)\
					*on_time 
	fan_power = '%0.1f' %fan_power_v


	#================================< Plotting >================================#

	plt.rcParams["figure.figsize"] = (20,10)
	plt.rcParams['lines.linewidth'] = 3
	plt.rcParams['axes.grid'] = True
	plt.rcParams['font.family'] = 'Times New Roman'
	plt.rcParams['font.size'] = 25
	# plt.rcParams['figure.dpi'] = 500 # chk


	# Set xlabel
	hrs = np.arange(0, 24, 1)
	xlabel_hr = np.array([])

	if now[3] > end_time[3]:
		xlabel_hr = np.append(xlabel_hr, hrs[now[3]:])
		xlabel_hr = np.append(xlabel_hr, hrs[:end_time[3]])
	else:
		xlabel_hr = np.append(xlabel_hr, hrs[now[3]:end_time[3]])

	x_label_hour = [str(int(xlabel_hr[i])).zfill(2) + ':%s' %(str(int(now[4])).zfill(2)) for i in range(len(xlabel_hr))]


	# Set title
	title = '%s.%s.%s. ~ %s.%s.%s.' %(now[0], now[1], now[2], end_time[0], end_time[1], end_time[2])
	
	Result = {'title': title, 'future_num': future_num, 'x_label': x_label_hour, 'RA_asis': RA_asis,
			  'RA_tobe': RA_TOBE, 'RA_fail': RA_proposal[0],
			  'OA': OA, 'RA_proposal': RA_proposal,
			  'damp_on_best': damp_on_best, 'Damper_fail': damp_asis[0], 'damp_asis': damp_asis,
			  'damper_ratio': damp_ratio, 'AHU_on_proposal': AHU_on_proposal, 'load_saving': load_saving,
			  'on_idx': rest_idx, 'msg': 'Success - Done'}
	return Result
	
	'''
	# Plot
	fig = plt.figure() 
	gs = gridspec.GridSpec(nrows=2, ncols=1,
							height_ratios=[2.5, 1])

	ax0 = plt.subplot(gs[0])
	ax0.plot(RA_asis, color = 'k', ls = '--', label = '$T_{RA}$  (As-Is)') 
	ax0.plot(RA_TOBE, color = 'r', label = '$T_{RA}$ (To-Be)', marker = 'o', markersize = 10, mfc='none')
	ax0.plot(RA_proposal[0], ls = '--', lw = 1, color = 'k', label = 'Fail')
	[ax0.plot(RA_proposal[i], ls = '--', lw = 1, color = 'k') for i in range(1 , len(RA_proposal))]
	ax0.plot(OA, color = 'y', label = '$T_{OA}$')
	ax0.set_xticks(np.arange(0, future_num, 4))
	ax0.set_xticklabels([])
	ax0.set_xlim([0, future_num-1])
	ax0.legend()
	ax0.set_ylabel('Temperature ($^\circ$C)')
	ax0.set_title(title)

	ax1 = plt.subplot(gs[1])
	ax1.plot(damp_on_best, color = 'r', label = 'To-Be')
	ax1.plot(damp_asis[0], ls = '--', lw = 1, color = 'k', label = 'Fail')
	[ax1.plot(damp_asis[i], ls = '--', lw = 1, color = 'k') for i in range(1 , len(AHU_on_proposal))]
	ax1.set_xticks(np.arange(0, future_num, 4))
	ax1.set_xticklabels(x_label_hour)
	ax1.set_xlim([0, future_num-1])
	ax1.set_ylabel('Damper open rate (%)') # chk
	ax1.set_ylim(damp_ratio) # chk
	ax1.set_yticks(damp_ratio)
	ax1.legend()
	ax1.set_yticklabels(['Off', 'On'])    

	plt.show()

	print('on_idx',on_idx)
	'''