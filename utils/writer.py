############################################################
# Module  : Trace program execution and print to screen
# Date    : March 23rd, 2017
# Author  : Xiao Ling
############################################################

import os
import datetime

############################################################
'''
	Class
'''
class Writer:
	'''
		@Use: construct writer instance to print to screen
		      and write output to `log_dir`
		      with tabs `tabs` preceding each line
		      if function is in `debug` mode. 
		      print `debug mode on screen`

		@Returns: a `Writer` instance

		@Methods:

			tell  :: String -> IO ()
				prints message to screen and write 
				message to output file

				Throws: ValueError: if Handle is closed

			close :: IO ()
				closes IO Handle

	'''
	def __init__(self, log_dir, tabs=0, debug = False, console = True):
		if not os.path.exists(log_dir):
			raise NameError("\n\t>> no directory at " + log_dir)
		else:
			files = os.listdir(log_dir)
			name      = 'log-' + str(len(files) + 1) + '.txt'
			self.path = os.path.join(log_dir,name)
			self.tabs = '\t'*tabs + '>> '
			self.console = console

			if debug:
				mode = ' in DEBUG mode '
			else:
				mode = ' in PRODUCTION mode '

			with open(self.path,'a') as h:
				self.tell('='*10 + mode + '='*10)



	def tell(self,msg):
		if self.console:
			print('\n' + self.tabs + msg)

		with open(self.path,'a') as h:
			h.write('\n' + self.tabs + msg)







