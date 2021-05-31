from datetime import date, datetime, timedelta


MONTH = ('Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
DAYS_IN_MONTH = (31, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

TS = timedelta(minutes=15)


def now(now='now'):
	if isinstance(now, tuple):
		now_dt = datetime(*now)
	elif isinstance(now, datetime):
		now_dt = now
	elif isinstance(now, str):
		now_dt = datetime.now()
	return datetime(now_dt.year, now_dt.month, now_dt.day, now_dt.hour, (now_dt.minute//15)*15)


def week_start(d, weekday):
	day = date(*d[:3])
	return date_to_tuple(day - timedelta(days=1)*abs(weekday-day.weekday()))


def time_to_string(year, month, day, hour=0, minute=0):
	return f'{year}-{month:02}-{day:02} {hour:02}:{minute:02}:00'


def time_to_string_sql(year, month, day, hour=0, minute=0):
	return f'{year}-{month:02}-{day:02} {hour:02}:{minute:02}:00.000'


def string_to_time(string):
	time_tuple = (
		int(string[0:4]), int(string[5:7]), int(string[8:10]),
		int(string[11:13]), int(string[14:16])#, int(string[17:19])
	)
	return time_tuple


def date_to_tuple(d):
	return (d.year, d.month, d.day)


def datetime_to_tuple(dt):
	return (dt.year, dt.month, dt.day, dt.hour, (dt.minute//15)*15)


def datetime_to_string(dt):
	return time_to_string(dt.year, dt.month, dt.day, dt.hour, (dt.minute//15)*15)


def time_idx_bound(time, info):
	'''원하는 기간의 시간 index를 반환하는 함수'''

	if isinstance(info, int):
		delta = TS * abs(info)

		if info > 0:
			time_1 = time
			time_2 = datetime_to_tuple(datetime(*time)+delta)
		elif info < 0:
			time_2 = datetime_to_tuple(datetime(*time))
			time_1 = datetime_to_tuple(datetime(*time_2)-delta)

	if isinstance(info, tuple):
		if datetime(*time) > datetime(*info):
			temp = time
			time = info
			info = temp
		time_1 = time
		time_2 = datetime_to_tuple(datetime(*info) - TS)

	time_string_1 = time_to_string(*time_1)
	time_string_2 = time_to_string(*time_2)

	return time_string_1, time_string_2


def time_range(time, info):

	time_string_1, time_string_2 = time_idx_bound(time, info)
	time_1 = datetime(*string_to_time(time_string_1))
	time_2 = datetime(*string_to_time(time_string_2))
	duration = int((time_2 - time_1) / TS)

	return [datetime_to_tuple(time_1 + TS*idx) for idx in range(duration)]


def time_idx_range(time, info):
	return [time_to_string(*_) for _ in time_range(time, info)]


#* For model generation

'''
def train_start_dt(generated, bounds, is_cool):
	year = generated.year
	bound_1 = datetime(year, *bounds[0])
	bound_2 = datetime(year, *bounds[1])
	if is_cool:
		if bound_1 - generated < timedelta(days=15):
			return datetime(year-1, *bounds[0])
		else:
			return bound_1
	else:
		pass
'''
def train_start_dt(generated, bounds, is_cool, pad_days=25):
	year = generated.year

	if is_cool:
		#? 냉방
		bound = bounds[0]
	else:
		#? 난방
		bound = bounds[1]
	
	if generated < datetime(year, *bound) + timedelta(days=pad_days):
		return datetime(year-1, *bound)
	else:
		return datetime(year, *bound)


#* For model update

def year_segment(dt, bounds):
	bound_1 = datetime(dt.year, *bounds[0])
	bound_2 = datetime(dt.year, *bounds[1])
	print('dt, bound_1, bound_2', dt, bound_1, bound_2)
	if dt < bound_1: return 0
	elif dt < bound_2: return 1
	else: return 2


def generate_update_queue(model_dt, now_dt, cool, bounds):

	model_d = datetime(model_dt.year, model_dt.month, model_dt.day)
	now_d = datetime(now_dt.year, now_dt.month, now_dt.day)

	if now_d <= model_d: return [], model_dt

	update_queue = []
	update_dt = now_dt

	model_tpl = datetime_to_tuple(model_dt)
	now_tpl = datetime_to_tuple(now_dt)
	now_tpl_wo_year = (now_dt.month, now_dt.day, now_dt.hour, now_dt.minute)

	model_seg = year_segment(model_dt, bounds)
	now_seg = year_segment(now_dt, bounds)
	print('model_seg',model_seg)
	print('now_seg',now_seg)
	print(model_dt.year, now_dt.year)
	if cool:
		#* 냉방

		#? 당해년도
		if model_seg == 0:
			if now_seg == 2 or model_dt.year < now_dt.year:
				update_queue.append((
					(model_dt.year, *bounds[0]),
					(model_dt.year, *bounds[1])
				))
			elif now_seg == 1:
				update_queue.append((
					(model_dt.year, *bounds[0]),
					(model_dt.year, *now_tpl_wo_year)
				))
		elif model_seg == 1:
			if now_seg == 2 or model_dt.year < now_dt.year:
				update_queue.append((
					model_tpl,
					(model_dt.year, *bounds[1])
				))
			elif now_seg == 1:
				update_queue.append((
					model_tpl,
					now_tpl
				))

		#? 다음해
		if model_dt.year < now_dt.year:

			for yr in range(model_dt.year+1, now_dt.year):
				update_queue.append((
					(yr, *bounds[0]),
					(yr, *bounds[1])
				))
			
			if now_seg == 1:
				update_queue.append((
					(now_dt.year, *bounds[0]),
					now_tpl
				))
			elif now_seg == 2:
				update_queue.append((
					(now_dt.year, *bounds[0]),
					(now_dt.year, *bounds[1])
				))
	
	else:
		#* 난방

		#? 당해년도
		if model_seg == 0:
			if model_dt.year < now_dt.year:
				update_queue.append((
					model_tpl,
					(model_dt.year, *bounds[0])
				))
				update_queue.append((
					(model_dt.year, *bounds[1]),
					(model_dt.year+1, 1, 1)
				))
			elif now_seg == 0:
				update_queue.append((
					model_tpl,
					now_tpl
				))
			elif now_seg == 1:
				update_queue.append((
					model_tpl,
					(model_dt.year, *bounds[0])
				))
			elif now_seg == 2:
				update_queue.append((
					model_tpl,
					(model_dt.year, *bounds[0])
				))
				update_queue.append((
					(model_dt.year, *bounds[1]),
					now_tpl
				))
		elif model_seg == 1:
			if model_dt.year < now_dt.year:
				update_queue.append((
					(model_dt.year, *bounds[1]),
					(model_dt.year+1, 1, 1)
				))
			elif now_seg == 2:
				update_queue.append((
					(model_dt.year, *bounds[1]),
					now_tpl
				))
		elif model_seg == 2:
			if model_dt.year < now_dt.year:
				update_queue.append((
					model_tpl,
					(model_dt.year+1, 1, 1)
				))
			elif now_seg == 2:
				update_queue.append((
					model_tpl,
					now_tpl
				))

		#? 다음해
		if model_dt.year < now_dt.year:

			for yr in range(model_dt.year+1, now_dt.year):
				update_queue.append((
					(yr, 1, 1),
					(yr, *bounds[0])
				))
				update_queue.append((
					(yr, *bounds[1]),
					(yr+1, 1, 1)
				))
			print('ii',now_seg)
			if now_seg == 0:
				update_queue.append((
					(now_dt.year, 1, 1),
					now_tpl
				))
			elif now_seg == 1:
				update_queue.append((
					(now_dt.year, 1, 1),
					(now_dt.year, *bounds[0])
				))
			elif now_seg == 2:
				update_queue.append((
					(now_dt.year, 1, 1),
					(now_dt.year, *bounds[0])
				))
				update_queue.append((
					(now_dt.year, *bounds[1]),
					now_tpl
				))

	for q in update_queue:
		print(q)
	return update_queue, update_dt


if __name__ == '__main__':
	'''
	update_queue, _ = generate_update_queue(
		datetime(2020, 8, 2),
		datetime(2020, 10, 3),
		True,
		((5,11), (10,25))
	)
	for q in update_queue:
		print(q)

	print()

	update_queue, _ = generate_update_queue(
		datetime(2019, 4, 1),
		datetime(2021, 8, 21, 12),
		False,
		((5,11), (10,25))
	)
	for q in update_queue:
		print(q)
	'''
	# print(time_idx_range((2020, 7, 1, 14), -5))

	# print(week_start((2021, 1, 1), -1))

	# print(now())

	# print(train_start_dt(datetime(2021, 1, 26), ((5,11), (10,25)), False))

