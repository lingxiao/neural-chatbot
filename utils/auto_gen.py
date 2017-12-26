############################################################
# Module  : Auto generate python file
# Date    : April 2nd, 2017
# Author  : Xiao Ling
############################################################

import os
import copy
from utils import *

'''
	@Use: given `src_path` to existing foo.py and nonexisting bar.py
		  open foo.py and swap out `src_str` string in foo.py
		  for `tgt_str` string in bar.py
		  write to `tgt_path`
'''
def auto_gen(src_path, tgt_path, src_str, tgt_str):

	src = open(src_path,'rb').read().split('\n')
	tgt = [swap(x, src_str, tgt_str) for x in src]

	with open(tgt_path,'wb') as h:
		h.write('\n'.join(tgt))


def swap(xs, src_str, tgt_str):
	if src_str in xs:
		return xs.replace(src_str, tgt_str)
	else:
		return xs

