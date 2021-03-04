import datetime as dt


class Timer:

	def __init__(self):
		self.start_dt = dt.datetime.now()

	def stop(self):
		print('Time taken: %s' % (dt.datetime.now() - self.start_dt))
