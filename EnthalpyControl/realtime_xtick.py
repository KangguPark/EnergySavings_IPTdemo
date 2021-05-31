def fn_realtime_xtick(hour_f,minute_f):
    import copy
    if minute_f <15:
        minute_f =15
    elif minute_f >14 and minute_f <30:
        minute_f=30
    elif minute_f>29 and minute_f <45:
        minute_f=45
    elif minute_f>44 and minute_f <60:
        minute_f=0

    equal=":"
    minute_zero="00"
    # 3시간전
    # 1시간 
    if minute_f==0:
        hour_1 = str(hour_f-3) + equal +minute_zero
    else: hour_1 = str(hour_f-3) + equal +str(minute_f)
    
    # 2시간 45분전
    if minute_f == 0:
        hour_2 = str(hour_f-3) + equal + str(15)
    elif minute_f==45:
        hour_2 =  str(hour_f-2) + equal + minute_zero
    else: hour_2 = str(hour_f-3) + equal + str(minute_f+15)
    
    # 2시간 30분전
    if minute_f == 0:
        hour_3 = str(hour_f-3) + equal + str(30)
    elif minute_f==30:
        hour_3 = str(hour_f-2) + equal + minute_zero
    elif minute_f==45:
        hour_3= str(hour_f-3) +equal +str(minute_f-30)
    else: hour_3 = str(hour_f-3) + equal + str(minute_f+30)
    
    # 2시간 15분전
    if minute_f == 0:
        hour_4 = str(hour_f-3) + equal + str(45)
    elif minute_f == 15:
        hour_4 = str(hour_f-2) +equal + minute_zero
    
    else: hour_4 = str(hour_f-2) + equal + str(minute_f-15)
    
    # 1시간 
    if minute_f==0:
        hour_5 = str(hour_f-2) + equal +minute_zero
    else: hour_5 = str(hour_f-2) + equal +str(minute_f)
    
    
    # 1시간 45분전
    if minute_f == 0:
        hour_6 = str(hour_f-2) + equal + str(15)
    elif minute_f==45:
        hour_6 =  str(hour_f-1) + equal + minute_zero
    else: hour_6 = str(hour_f-2) + equal + str(minute_f+15)
    
    # 1시간 30분전
    if minute_f == 0:
        hour_7 = str(hour_f-2) + equal + str(30)
    elif minute_f==30:
        hour_7 = str(hour_f-1) + equal + minute_zero
    elif minute_f==45:
        hour_7= str(hour_f-1) +equal +str(minute_f-30)
    else: hour_7 = str(hour_f-2) + equal + str(minute_f+30)
    
    # 1시간 15분전
    if minute_f == 0:
        hour_8 = str(hour_f-2) + equal + str(45)
    elif minute_f == 15:
        hour_8 = str(hour_f-1) +equal + minute_zero
    else: hour_8 = str(hour_f-1) + equal + str(minute_f-15)
    
    # 1시간 
    if minute_f==0:
        hour_9 = str(hour_f-1) + equal +minute_zero
    else: hour_9 = str(hour_f-1) + equal +str(minute_f)
    # 45분전
    if minute_f == 0:
        hour_10 = str(hour_f-1) + equal + str(15)
    elif minute_f==45:
        hour_10 =  str(hour_f) + equal + minute_zero
    else: hour_10 = str(hour_f-1) + equal + str(minute_f+15)
    
    # 30분전
    if minute_f == 0:
        hour_11 = str(hour_f-1) + equal + str(30)
    elif minute_f==30:
        hour_11 = str(hour_f) + equal + minute_zero
    elif minute_f==45:
        hour_11= str(hour_f) +equal +str(minute_f-30)
    else: hour_11 = str(hour_f-1) + equal + str(minute_f+30)
    
    # 15분전
    if minute_f == 0:
        hour_12 = str(hour_f-1) + equal + str(45)
    elif minute_f== 15:
        hour_12 = str(hour_f) +equal +str(minute_zero)
    else: hour_12 = str(hour_f) + equal + str(minute_f-15)
    
    if minute_f==0:
        hour_13 = str(hour_f) + equal +minute_zero
    else: hour_13 = str(hour_f) + equal +str(minute_f)
    
    if minute_f ==45:
        hour_14 = str(hour_f+1) +equal + minute_zero
    else:
        hour_14=str(hour_f) +equal +str(minute_f+15)
        
    if minute_f == 30:
        hour_15 = str(hour_f+1) +equal + minute_zero
    elif minute_f == 45:
        hour_15 = str(hour_f+1) +equal + str(minute_f-30)
    else:
        hour_15=str(hour_f) + equal + str(minute_f+30)
        
    if minute_f == 15:
        hour_16 = str(hour_f+1) +equal + minute_zero
    elif minute_f == 30:
        hour_16 = str(hour_f+1) + equal +str(minute_f-15)
    elif minute_f ==45:
        hour_16= str(hour_f+1) + equal +str(minute_f-15)
    else:
        hour_16=str(hour_f) +equal +str(minute_f+45)
    label = [hour_1, hour_2, hour_3, hour_4, hour_5, hour_6, hour_7, hour_8, hour_9, hour_10, hour_11, hour_12, hour_13, hour_14, hour_15, hour_16]
    return label
