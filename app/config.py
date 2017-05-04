############################################################
# Module  : All System Paths
# Date    : March 23rd, 2017
# Author  : Xiao Ling
############################################################

import os
import shutil
from utils import Writer


############################################################
'''
	Global vocabulary parameters
'''
RESERVED_TOKENS = {'pad'       : '_'
	    	      , 'unk'      : '<unk>'
			      , 'go'       : '<go>'
			      , 'eos'      : '</s>'    # end of sentence
			      , 'eoc'      : '</c>'    # end of conversation
			      ,'vocab-size': 50005}


############################################################
'''
	System Root
'''
root      = os.getcwd()

# local
if root[0:6] == '/Users':
	input_dir = '/Users/lingxiao/Documents/research/code/neural-chatbot-data'
	chkpt_dir = '/Users/lingxiao/Documents/research/code/neural-chatbot-checkpoint/'

# nlpgrid
elif root[0:5] == '/mnt/':
	input_dir = '/home1/l/lingxiao/xiao/neural-chatbot-data'
	chkpt_dir = '/home1/l/lingxiao/xiao/neural-chatbot/neural-chatbot-checkpoint/'

# tesla
elif root[0:5] == '/home':
	input_dir = '/data2/xiao/neural-chatbot-data'
	chkpt_dir = '/data2/xiao/neural-chatbot-checkpoint'

############################################################
'''
	System Environment
'''
PATH = {# directories
        'log'         : os.path.join(root, 'deploy/logs')
       ,'root'        : root

       ,'input'       : input_dir
       ,'checkpoint'  : chkpt_dir
  
       # movie corpus
       ,'movie': {
       	    'zip'             : os.path.join(input_dir, 'movie/zip'        )
       	   ,'raw'             : os.path.join(input_dir, 'movie/raw'        )

	       ,'sess-normed'     : os.path.join(input_dir, 'movie/sess-normed')
	       ,'sess-idx'        : os.path.join(input_dir, 'movie/sess-idx'    )
	       ,'sess-concat'     : os.path.join(input_dir, 'movie/sess-concat' )

       	   ,'dev'             : os.path.join(input_dir, 'movie/dev'        )
	       ,'sess-normed'     : os.path.join(input_dir, 'movie/dev/sess-normed')
	       ,'sess-idx-dev'    : os.path.join(input_dir, 'movie/dev/sess-idx')
	       ,'sess-concat-dev' : os.path.join(input_dir, 'movie/dev/sess-concat' )

	       ,'w2idx'           : os.path.join(input_dir, 'movie/w2idx.pkl'   )
		   ,'idx2w'           : os.path.join(input_dir, 'movie/idx2w.pkl'   )
	    }

       # phone corpus
       ,'phone': {
       	    'raw'             : os.path.join(input_dir, 'phone-home/raw'         )
	       ,'sess-normed'     : os.path.join(input_dir, 'phone-home/sess-normed' )
	       ,'sess-concat'     : os.path.join(input_dir, 'phone-home/sess-concat' )
	       ,'sess-idx'        : os.path.join(input_dir, 'phone-home/sess-idx'    )
	       ,'w2idx'           : os.path.join(input_dir, 'phone-home/w2idx.pkl'   )
		   ,'idx2w'           : os.path.join(input_dir, 'phone-home/idx2w.pkl'   )
	    }

	    # models
	    ,'model': {'hred' : os.path.join(root, 'models/ts_hred')}
	}


############################################################
'''
  @Use: given name `path`, locate full path if it exists
'''
def get_path(path):

	path = path.split('/')

	if len(path) == 1:

		path = path[0]
		if path in PATH and type(PATH[path]) == str:
			return PATH[path]
		else:
			return ''

	elif len(path) == 2:
		dirs, path = path

		if dirs in PATH:
			if path in PATH[dirs]:
				return PATH[dirs][path]
			else:
				return ''
		else:
			return ''
	else: 
		return ''

def setup():

	os.system('clear')

	log_dir = get_path('log')

	'''	
		reset log directories
	'''
	if not os.path.exists(log_dir):
		os.mkdir(log_dir)
	else:
	    shutil.rmtree(log_dir)
	    os.mkdir(log_dir)

	writer = Writer(log_dir)
	writer.tell('Initializing application [ neural-chatbot ] ...')

	for key,dirs in PATH.iteritems():

		if type(dirs) == str:
			if os.path.exists(dirs):
				writer.tell('path already exist at ' + dirs)
			else:
				writer.tell('creating path at ' + dirs)
				os.mkdir(dirs)

		elif type(dirs) == dict:

			for k,d in dirs.iteritems():
				if os.path.exists(d) \
				or '.txt' in d or '.pkl' in d or '.npy' in d:
					writer.tell('path already exist at ' + d)
				else:
					writer.tell('creating path at ' + d)
					os.mkdir(d)


	writer.tell('complete application setup!')

############################################################
'''
	setup project path 
'''
setup()


