# # Data accumulator 
# 
# 참고 : https://signing.tistory.com/22
# 공공데이터포털 : https://www.data.go.kr/
# 본 파일은 동네예보 조회서비스 중 동네예보조회에 대한 내용임
# 공공데이터포털에서 제공하는 동네예보 조회서비스 API는 최근 1일까지의 데이터만 제공하고 있음

from urllib.request import urlopen
from urllib.parse import urlencode, unquote, quote_plus
from datetime import datetime, timedelta
import urllib
import requests
import json
import pandas as pd
import pymssql
import configparser
import time
import numpy as np
# %%
def get_WF_Temperature_Humidity_info(data):
	HumWF=[]
	TempWF=[]
	try: 
		weather_info = data['response']['body']['items']['item'] 
		for i in range(len(weather_info)):
			if weather_info[i]['category'] == 'T3H':
				TempWF.append([weather_info[i]['baseDate'], weather_info[i]['baseTime'], weather_info[i]['fcstDate'], weather_info[i]['fcstTime'],weather_info[i]['fcstValue']])
			elif weather_info[i]['category'] == 'REH':
				HumWF.append([weather_info[i]['baseDate'], weather_info[i]['baseTime'], weather_info[i]['fcstDate'], weather_info[i]['fcstTime'],weather_info[i]['fcstValue']])
		return TempWF, HumWF
	except KeyError:
		print('API 호출 실패! (Weather forecast data)')

def get_base_time(hour): 
    hour = int(hour)
    if hour < 3: 
        temp_hour = '20' 
    elif hour < 6: 
        temp_hour = '23' 
    elif hour < 9: 
        temp_hour = '02' 
    elif hour < 12: 
        temp_hour = '05' 
    elif hour < 15: 
        temp_hour = '08'
    elif hour < 18:
        temp_hour = '11' 
    elif hour < 20: 
        temp_hour = '14' 
    elif hour < 24: 
        temp_hour = '17' 
    return temp_hour + '00'

def get_weather_forecast(n_x, n_y): 
	now = datetime.now()
	now_date = now.strftime('%Y%m%d')
	now_hour = int(now.strftime('%H'))
	if now_hour < 6: 
		base_date = str(int(now_date) - 1)
	else: 
		base_date = now_date
	base_hour = get_base_time(now_hour)

	num_of_rows = '90'
	base_date = base_date
	base_time = base_hour
#	base_date = '20200622'
#	base_time = '1700'
	# 해당 지역에 맞는 죄표 입력
		
	# Setting for URL parsing
	CallBackURL = 'http://apis.data.go.kr/1360000/VilageFcstInfoService/getVilageFcst'		# 맨 마지막 명칭에 따라 상세기능에 대한 정보가 변경될 수 있음
																							# getUltraSrtNcst: 초단기실황조회, getUltraSrtFcst: 초단기예보조회, getVilageFcst: 동네예보조회, getFcstVersion: 예보버전조회
	params = '?' + urlencode({
		quote_plus("serviceKey"): "TzI4xbYubSKs4S9l6n1cQPjfHw1W4hIMqTEpVhSmX5AZx4oi6FE%2FjeIR2lqTOvVMR1ReGMdTAf3d3NsDlzI%2B1Q%3D%3D",  # 인증키 (2년마다 갱신 필요)  # 반드시 본인이 신청한 인증키를 입력해야함 (IP 불일치로 인한 오류 발생 가능)
		quote_plus("numOfRows"): num_of_rows,          # 한 페이지 결과 수 // default : 10
		quote_plus("pageNo"): "1",              # 페이지 번호 // default : 1
		quote_plus("dataType"): "JSON",         # 응답자료형식 : XML, JSON
		quote_plus("base_date"): base_date,    # 발표일자 // yyyymmdd
		quote_plus("base_time"): base_time,        # 발표시각 // HHMM, 매 시각 40분 이후 호출
		quote_plus("nx"): n_x,                # 예보지점 X 좌표
		quote_plus("ny"): n_y                 # 예보지점 Y 좌표
	})
	# URL parsing
	req = urllib.request.Request(CallBackURL + unquote(params))

	# Get Data from API
	response_body = urlopen(req).read() # get bytes data

	# Convert bytes to json
	json_data = json.loads(response_body)
	# Every result
	res = pd.DataFrame(json_data['response']['body']['items']['item'])
	#print('\n============================== Result ==============================')
	#print(res)
	#print('=====================================================================\n')
	TemperatureWF, HumidityWF = get_WF_Temperature_Humidity_info(json_data) 
	
	return TemperatureWF, HumidityWF


def get_Temperature_Humidity_info(data, len):
    temperature=[]
    humidity=[]
	
    try:     
        weather_info = data['response']['body']['items']['item'] 
        for i in range(int(len)):
            temperature.append(weather_info[i]['ta'])
            humidity.append(weather_info[i]['hm'])
        return temperature, humidity
    except KeyError:
        print('API 호출 실패! (Actual weather data)')

def get_weather(start_day, end_day):
    if end_day.hour > start_day.hour:
        hour_term = (end_day - start_day).seconds/3600
        day_term = (end_day - start_day).days
        UnkonwDataLen = day_term*24 + hour_term
    else:
        hour_term = (start_day - end_day).seconds/3600
        day_term = (end_day - start_day).days +1
        UnkonwDataLen = day_term*24 - hour_term
        
    sYear = str(start_day.year)
    if start_day.month < 10:
        sMonth = "0" + str(start_day.month)
    else:
        sMonth=str(start_day.month)
    if start_day.day < 10:
        sDay = "0" + str(start_day.day)
    else:
        sDay = str(start_day.day)    
    start_date = sYear + sMonth + sDay
    if start_day.hour < 10:
        sTime = "0" + str(start_day.hour)
    else:
        sTime = str(start_day.hour)        
    start_time = sTime
    
    eYear = str(end_day.year)
    if end_day.month < 10:
        eMonth = "0" + str(end_day.month)
    else:
        eMonth=str(end_day.month)
    if end_day.day < 10:
        eDay = "0" + str(end_day.day)
    else:
        eDay = str(end_day.day)    
    end_date = eYear + eMonth + eDay
    if end_day.hour < 10:
        eTime = "0" + str(end_day.hour)
    else:
        eTime = str(end_day.hour)
    end_time = eTime
	
    # Setting for URL parsing
    CallBackURL = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList'      # 맨 마지막 명칭에 따라 상세기능에 대한 정보가 변경될 수 있음
    #   parameter for request
    params = '?' + urlencode({
        quote_plus("serviceKey"): "TzI4xbYubSKs4S9l6n1cQPjfHw1W4hIMqTEpVhSmX5AZx4oi6FE%2FjeIR2lqTOvVMR1ReGMdTAf3d3NsDlzI%2B1Q%3D%3D",     # 인증키  # 반드시 본인이 신청한 인증키를 입력해야함 (IP 불일치로 인한 오류 발생 가능)
        quote_plus("numOfRows"): str(int(UnkonwDataLen)),          # 한 페이지 결과 수 // default : 10
        quote_plus("pageNo"): "1",              # 페이지 번호 // default : 1
        quote_plus("dataType"): "JSON",         # 응답자료형식 : XML, JSON
        quote_plus("dataCd"): "ASOS",
        quote_plus("dateCd"): "HR",         # 날짜 분류 코드: DAY, HR
        quote_plus("startDt"): start_date,    # 시작일 // yyyymmdd
        quote_plus("startHh"): start_time,        # 시작시 // HH
        quote_plus("endDt"): end_date,    # 종료일 // yyyymmdd
        quote_plus("endHh"): end_time,        # 종료시 // HH
        quote_plus("stnIds"): "143",                # 지점번호 대구: 143
        quote_plus("schListCnt"): "10"
    })
    # URL parsing
    req = urllib.request.Request(CallBackURL + unquote(params))

    # Get Data from API
    response_body = urlopen(req).read() # get bytes data
    # Convert bytes to json
    json_data = json.loads(response_body)
    # Every result
    res = pd.DataFrame(json_data['response']['body']['items']['item'])
    #print('\n============================== Result ==============================')
    #print(res)
    #print('=====================================================================\n')
    Temperature, Humidity = get_Temperature_Humidity_info(json_data, UnkonwDataLen)
    return Temperature, Humidity
	
def Check_Restoring_Unknown_past_data(targetDB_IP, targetDB_UserID, targetDB_UserPW, targetDB_Name, n_x, n_y):
	# MSSQL Access
	conn = pymssql.connect(host = targetDB_IP, user = targetDB_UserID, password = targetDB_UserPW, database = targetDB_Name)
	# Create Cursor from Connection
	cursor = conn.cursor()
	
	now = datetime.now()
	InitialForecastDayforCheck = now - timedelta(days=90)	# 최대 약 3개월 전까지 데이터 확인
	# API가 전날 데이터까지 제공함 (오늘 예보데이터가 없다면 다음날 채워야 함)
	FinalForecastDayforCheck = now
	
	## Temperature 가져오기
	cursor.execute("SELECT * FROM "+targetDBName+".dbo.BemsMonitoringPointWeatherForecasted where SiteId = 1 and Category="+"'"+"Temperature"+"'"+" and ForecastedDateTime >= "+"'"+str(InitialForecastDayforCheck.year)+"-"+str(InitialForecastDayforCheck.month)+"-"+str(InitialForecastDayforCheck.day)+"'"+"and ForecastedDateTime < "+"'"+str(FinalForecastDayforCheck.year)+"-"+str(FinalForecastDayforCheck.month)+"-"+str(FinalForecastDayforCheck.day)+"' order by ForecastedDateTime asc")
	
	# 데이타 하나씩 Fetch하여 출력
	row = cursor.fetchone()
	TemperatureRawData = [row]
	while row:
		row = cursor.fetchone()
		if row == None:
			break
		TemperatureRawData.append(row)
	
	## Humidity 가져오기
	cursor.execute("SELECT * FROM "+targetDBName+".dbo.BemsMonitoringPointWeatherForecasted where SiteId = 1 and Category="+"'"+"Humidity"+"'"+" and ForecastedDateTime >= "+"'"+str(InitialForecastDayforCheck.year)+"-"+str(InitialForecastDayforCheck.month)+"-"+str(InitialForecastDayforCheck.day)+"'"+"and ForecastedDateTime < "+"'"+str(FinalForecastDayforCheck.year)+"-"+str(FinalForecastDayforCheck.month)+"-"+str(FinalForecastDayforCheck.day)+"' order by ForecastedDateTime asc")
	
	# 데이타 하나씩 Fetch하여 출력
	row = cursor.fetchone()
	HumidityRawData = [row]
	while row:
		row = cursor.fetchone()
		if row == None:
			break
		HumidityRawData.append(row)
	
	conn.close()
	
	TimeIdx_3h_Interval = [datetime(InitialForecastDayforCheck.year, InitialForecastDayforCheck.month, InitialForecastDayforCheck.day, 0, 0, 0)]
	TimeIdx_Final = datetime(FinalForecastDayforCheck.year, FinalForecastDayforCheck.month, FinalForecastDayforCheck.day, 0, 0, 0)
	while TimeIdx_3h_Interval[-1] < TimeIdx_Final:
		TimeIdx_3h_Interval.append(TimeIdx_3h_Interval[-1]+timedelta(hours=3))
	TimeIdx_3h_Interval = TimeIdx_3h_Interval[0:-1]
	
	### DB에 비어있는 값 찾기
	idx_tem = 0								# TemperatureRawData 인덱스
	idx_hum = 0								# HumidityRawData 인덱스
	InitialDay_UnknownData_Tem = []			# 기온 unkown data 초기 일시
	FinalDay_UnkownDate_Tem = []			# 기온 unkown data 연속 종료 일시
	isContinue_Tem = False	
	InitialDay_UnknownData_Hum = []			# 습도 unkown data 초기 일시
	FinalDay_UnkownDate_Hum = []			# 습도 unkown data 연속 종료 일시
	isContinue_Hum = False
	
	for i in range(len(TimeIdx_3h_Interval)):
		# DB 마지막 일시의 데이터와 복원하고자하는 일시의 마지막 데이터가 일치하지 않는 경우(기온)
		if i >= len(TemperatureRawData):	
			if idx_tem >= len(TemperatureRawData):				## 복원하고자하는 데이터의 마지막 일시 할당 후 for문 종료
				InitialDay_UnknownData_Tem.append(TimeIdx_3h_Interval[i])
				FinalDay_UnkownDate_Tem.append(TimeIdx_3h_Interval[-1])		
				break
			elif TimeIdx_3h_Interval[i] == TemperatureRawData[idx_tem][4]:
				idx_tem += 1
				if isContinue_Tem == True:
					FinalDay_UnkownDate_Tem.append(TimeIdx_3h_Interval[i] - timedelta(hours=3))
				isContinue_Tem = False
			else:
				if isContinue_Tem == False:
					InitialDay_UnknownData_Tem.append(TimeIdx_3h_Interval[i])
				isContinue_Tem = True
		#####
		
		# DB에 최근 데이터가 있는 경우(기온)
		else:								
			if TimeIdx_3h_Interval[i] == TemperatureRawData[idx_tem][4]:
				idx_tem += 1
				if isContinue_Tem == True:
					FinalDay_UnkownDate_Tem.append(TimeIdx_3h_Interval[i] - timedelta(hours=3))
				isContinue_Tem = False
			else:
				if isContinue_Tem == False:
					InitialDay_UnknownData_Tem.append(TimeIdx_3h_Interval[i])
				isContinue_Tem = True
				
	for i in range(len(TimeIdx_3h_Interval)):
		# DB 마지막 일시의 데이터와 복원하고자하는 일시의 마지막 데이터가 일치하지 않는 경우(습도)
		if i >= len(HumidityRawData):	
			if idx_hum >= len(HumidityRawData):				## 복원하고자하는 데이터의 마지막 일시 할당 후 for문 종료
				InitialDay_UnknownData_Hum.append(TimeIdx_3h_Interval[i])
				FinalDay_UnkownDate_Hum.append(TimeIdx_3h_Interval[-1])		
				break
			elif TimeIdx_3h_Interval[i] == HumidityRawData[idx_hum][4]:
				idx_hum += 1
				if isContinue_Hum == True:
					FinalDay_UnkownDate_Hum.append(TimeIdx_3h_Interval[i] - timedelta(hours=3))
				isContinue_Hum = False
			else:
				if isContinue_Hum == False:
					InitialDay_UnknownData_Hum.append(TimeIdx_3h_Interval[i])
				isContinue_Hum = True
		#####
		
		# DB에 복원하고자하는 일시의 마지막 데이터가 있는 경우(습도)
		else:
			if TimeIdx_3h_Interval[i] == HumidityRawData[idx_hum][4]:
				idx_hum += 1
				if isContinue_Hum == True:
					FinalDay_UnkownDate_Hum.append(TimeIdx_3h_Interval[i] - timedelta(hours=3))
				isContinue_Hum = False
			else:
				if isContinue_Hum == False:
					InitialDay_UnknownData_Hum.append(TimeIdx_3h_Interval[i])
				isContinue_Hum = True
	
	### Restoring unknown data from actual past weather data
	# MSSQL Access
	conn = pymssql.connect(host = targetDB_IP, user = targetDB_UserID, password = targetDB_UserPW, database = targetDB_Name)
	# Create Cursor from Connection
	cursor = conn.cursor()	
	for i in range(len(FinalDay_UnkownDate_Tem)):
		Tem, Hum = get_weather(InitialDay_UnknownData_Tem[i], FinalDay_UnkownDate_Tem[i] + timedelta(hours=1))  	## API 특징 end_time은 포함하지않으므로
		tem_date = InitialDay_UnknownData_Tem[i]
		for j in range(0,len(Tem),3):
			try:
				cursor.execute("INSERT INTO " + targetDBName + ".dbo.BemsMonitoringPointWeatherForecasted (SiteId, CreatedDateTime, Category, BaseDateTime, ForecastedDateTime, ForecastedValue, nx, ny) VALUES(1," + "'" + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "','"+ "Temperature" + "','"+ str(tem_date) + "','" + str(tem_date) + "'," + str(Tem[j]) + "," + n_x + "," + n_y + ")")
				conn.commit()				
			except:
				print('There is an issue in the progress of restoring unknown weather forecast data to actual past weather data. (Temperature)')						
			tem_date += timedelta(hours=3)
	
	for i in range(len(FinalDay_UnkownDate_Hum)):
		Tem, Hum = get_weather(InitialDay_UnknownData_Hum[i], FinalDay_UnkownDate_Hum[i] + timedelta(hours=1))  	## API 특징 end_time은 포함하지않으므로
		hum_date = InitialDay_UnknownData_Hum[i]
		for j in range(0,len(Hum),3):
			try:
				cursor.execute("INSERT INTO " + targetDBName + ".dbo.BemsMonitoringPointWeatherForecasted (SiteId, CreatedDateTime, Category, BaseDateTime, ForecastedDateTime, ForecastedValue, nx, ny) VALUES(1," + "'" + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "','"+ "Humidity" + "','"+ str(hum_date) + "','" + str(hum_date) + "'," + str(Hum[j]) + "," + n_x + "," + n_y + ")")
				conn.commit()				
			except:
				print('There is an issue in the progress of restoring unknown weather forecast data to actual past weather data. (Humidity)')						
			hum_date += timedelta(hours=3)
	conn.close()
	
	if len(FinalDay_UnkownDate_Tem) == 0:
		print('There is no unknown data before (Temperature)')
	else:
		for i in range(len(InitialDay_UnknownData_Tem)):
			print('Initial and final date of unknown data (Tempearure) : ', InitialDay_UnknownData_Tem[i], FinalDay_UnkownDate_Tem[i])
	if len(FinalDay_UnkownDate_Hum) == 0:
		print('There is no unknown data before (Humidity)')
	else:
		for i in range(len(InitialDay_UnknownData_Hum)):
			print('Initial and final date of unknown data (Humidity) : ', InitialDay_UnknownData_Hum[i], FinalDay_UnkownDate_Hum[i])
	
	
if __name__ == "__main__" :

	targetDBIP = '61.33.215.50'
	targetDBUserID = 'DGB_SU'
	targetDBUserPW = 'dgbsu123'
	targetDBName = 'iBems_SU2'
		
	nx = str(89) 			# 예보지점 x 좌표
	ny = str(91) 			# 예보지점 y 좌표
	
	#### Check Unknown past data when starting program ####	
	Check_Restoring_Unknown_past_data(targetDBIP, targetDBUserID, targetDBUserPW, targetDBName, nx, ny)
	
	# Accumulate weather forecast data
	while True:		
		now = datetime.now()
		if now.hour == 1 and now.minute <= 15:				## 하루에 한번 오전 1시 이후에 과거 데이터 체크 (오전 1시는 임의로 정한 시각)
			Check_Restoring_Unknown_past_data(targetDBIP, targetDBUserID, targetDBUserPW, targetDBName, nx, ny)		## 과거 실 데이터가 어제까지만 제공되기 때문에 매일 어제 데이터가 있는지 체크
		if now.hour == 20 and now.minute >= 45 and now.minute <= 59:
			AccumulationActive = True
		else:
			AccumulationActive = False
			if now.minute > 55:
				print("[ Current Time -", now.hour,":", now.minute,":", now.second,"], " "Sleeping for 10 minutes... Accumulate weather forecasted data at 8:45 ~ 9:00 p.m. every day")
				time.sleep(60*10)
			else:
				print("[ Current Time -", now.hour,":", now.minute,":", now.second,"], " "Sleeping for 15 minutes... Accumulate weather forecasted data at 8:45 ~ 9:00 p.m. every day")
				time.sleep(60*15)
		
		if AccumulationActive:
							
			## 예보 데이터 얻을 수 있는 함수 #########################
			TempWF, HumWF = get_weather_forecast(nx, ny) 
			################################################
			
			# MSSQL Access
			conn = pymssql.connect(host = targetDBIP, user = targetDBUserID, password = targetDBUserPW, database = targetDBName)
			# Create Cursor from Connection
			cursor = conn.cursor()	
			
			for i in range(1, len(TempWF)):
				baseDate = TempWF[i][0]
				baseTime = TempWF[i][1]
				FcstDate = TempWF[i][2]
				FcstTime = TempWF[i][3]
				baseDateTime = datetime(int(baseDate[0:4]), int(baseDate[4:6]), int(baseDate[6:]), int(baseTime[0:2]), int(baseTime[2:]))
				FcstDateTime = datetime(int(FcstDate[0:4]), int(FcstDate[4:6]), int(FcstDate[6:]), int(FcstTime[0:2]), int(FcstTime[2:]))
				try:
					cursor.execute("INSERT INTO " + targetDBName + ".dbo.BemsMonitoringPointWeatherForecasted (SiteId, CreatedDateTime, Category, BaseDateTime, ForecastedDateTime, ForecastedValue, nx, ny) VALUES(1," + "'" + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "','"+ "Temperature" + "','"+ str(baseDateTime) + "','" + str(FcstDateTime) + "'," + TempWF[i][4] + "," + nx + "," + ny + ")")
					conn.commit()
				except:
					print('Weather forecasted temperature data already exists! (ForecastDateTime : '+str(FcstDateTime)+')')
			for i in range(1, len(HumWF)):
				baseDate = HumWF[i][0]
				baseTime = HumWF[i][1]
				FcstDate = HumWF[i][2]
				FcstTime = HumWF[i][3]
				baseDateTime = datetime(int(baseDate[0:4]), int(baseDate[4:6]), int(baseDate[6:]), int(baseTime[0:2]), int(baseTime[2:]))
				FcstDateTime = datetime(int(FcstDate[0:4]), int(FcstDate[4:6]), int(FcstDate[6:]), int(FcstTime[0:2]), int(FcstTime[2:]))
				try:
					cursor.execute("INSERT INTO " + targetDBName + ".dbo.BemsMonitoringPointWeatherForecasted (SiteId, CreatedDateTime, Category, BaseDateTime, ForecastedDateTime, ForecastedValue, nx, ny) VALUES(1," + "'" + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "','"+ "Humidity" + "','"+ str(baseDateTime) + "','" + str(FcstDateTime) + "'," + HumWF[i][4] + "," + nx + "," + ny + ")")
					conn.commit()
				except:					
					print('Weather forecasted humidity data already exists! (ForecastDateTime : '+str(FcstDateTime)+')')
				
			conn.close()

			print("Sleeping for 15 minutes ...")
			time.sleep(60*15)
			
			
			
			
			

