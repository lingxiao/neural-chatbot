############################################################
# Module  : split edges and make main-#.py
# Date    : April 2nd, 2017
# Author  : Xiao Ling
############################################################

import os
import json
import networkx as nx

import re

from utils   import *
from scripts import *
from app.config import PATH

############################################################
'''
	paths
'''
'''
	@Use: split edges into chunks to compute
	      weight on remote 
'''
def run_split(size):

	edges, words = load_as_list(gr_path)

	to_tuple     = lambda xs: (xs[0], xs[1])
	unique_edges = list(set( to_tuple(sorted([u,v])) for u in words for v in words ))
	splits       = list(chunks(unique_edges,size))

	cnt = 1
	
	for xs in splits:
		path = os.path.join(dedges, 'edge-' + str(cnt) + '.txt')
		with open(path,'wb') as h:
			for s,t in xs:
				h.write(s + ', ' + t + '\n')
		cnt += 1

	return cnt

'''
	@Use: rewrite main-#.py file
'''
def run_auto_main(tot):

	cnt = 2

	for k in xrange(tot - 2):
		src_path = os.path.join(dscripts, 'main-1.py')
		tgt_path = os.path.join(dscripts, 'main-' + str(cnt) + '.py')
		src_str  = 'batch = 1'
		tgt_str  = 'batch = ' + str(cnt)
		auto_gen(src_path, tgt_path, src_str, tgt_str)
		cnt += 1

'''
	@Use: rewrite main-#.sh file
'''
def run_auto_sh(tot):

	cnt = 2

	for k in xrange(tot - 2):
		src_path = os.path.join(dscripts,'main-1.sh')
		tgt_path = os.path.join(dscripts,'main-' + str(cnt) + '.sh')
		src_str  = 'main-1'
		tgt_str  = 'main-' + str(cnt)

		auto_gen(src_path, tgt_path, src_str, tgt_str)

		cnt +=1

