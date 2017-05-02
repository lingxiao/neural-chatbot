############################################################
# Module  : preprocess movie data  
# Date    : Febuary 28th
# Author  : Xiao Ling  
############################################################

import os
import re
import shutil
import zipfile
from app import *
from utils import *

############################################################

'''
    @Use: given path too zip directory and path to raw directory
          extract all zip files into raw directory
          remove non english files
'''
def extract_movie(zip_root = '', raw_root = ''):

    writer = Writer(config.get_path('log'), 1)

    writer.tell('removing previously extracted files')
    shutil.rmtree(raw_root)
    os.mkdir(raw_root)

    writer.tell('extracting all files ...')
    paths    = [os.path.join(zip_root,q) for q in os.listdir(zip_root) if '.DS_Store' not in q]

    '''
        extracting all files
    '''
    for path in paths:
        with zipfile.ZipFile(path,"r") as zip_ref:
            zip_ref.extractall(raw_root)

    writer.tell('removing all non-english files')

    paths = [os.path.join(raw_root,q) for q in os.listdir(raw_root) \
            if '.DS_Store' not in q and '.zip' not in q]

    for path in paths:
        ext = path.split("-")[-1]
        _,en = ext.split('.')
        if en != 'en':
            os.remove(path)

############################################################
'''
    @Use: Given:

            - directory to input raw phone conversations
            - path to normalized conversations directory 

          * open all files and normalize by:
            - removing all meta-information demarked by {...}
            - fold speaker turns

'''
def preprocess_movie(raw_root = '', out_root = ''):

    writer = Writer(config.get_path('log'), 1)

    writer.tell('running preprocessing movie by adding speaker to all sessions' \
               + ' and removing meta tokens ...')

    writer.tell('removing previously preprocessed files')
    shutil.rmtree(out_root)
    os.mkdir(out_root)

    raw_paths = [os.path.join(raw_root, p) for p in os.listdir(raw_root) if 'DS_Store' not in p]

    cnt = 1

    for path in raw_paths:
        out = os.path.join(out_root, 'sess-' + str(cnt) + '.txt')

        raw = remove_meta(add_round(path))

        with open(out,'wb') as h:
            for spkr,utter in raw:
                h.write(spkr + ': ' + utter + '\n')

        cnt +=1 


############################################################
'''
    @Use: add speaker id to consecutive rounds
'''
def add_round(in_path):

    rounds = []

    spkr = 'A'

    with open(in_path,'rb') as h:
        for line in h:
            line = line.replace('\n','')
            rounds.append((spkr, line))
            if spkr == 'A': spkr = 'B'
            else: spkr = 'A'

    return rounds


'''
    @Use: remove all words in brackets and parensethis, ie:
            (...)
            [...]
            {...}
            <..>
            </..>

        - if any round is now the empty string, remove it
        - if there are consecutive rounds by the same speaker
            fold them into one round

    @Input: raw :: [(String, String)]   
    @output     :: [(String, String)]   
'''
def remove_meta(raw):

    '''
        strip rounds of brackets and only keep non empty rounds
    '''
    stripped = [ (s,rmv_meta(u)) for s,u in raw \
               if rmv_meta(u) ]


    '''
        fold consecutive rounds
    '''
    folded = [stripped[0]]

    for spkr,utter in stripped[1:]:
        prev_spkr, prev_utter = folded[-1]
        if spkr == prev_spkr:
            folded[-1] = (spkr, prev_utter + ' ' + utter)
        else:
            folded.append((spkr,utter))

    return folded


def rmv_meta(utter):

    utter = re.sub(r'.*?\((.*?)\)', '', utter)
    utter = re.sub(r'.*?\[(.*?)\]', '', utter)
    utter = re.sub(r'.*?\{(.*?)\}', '', utter)
    u = utter.replace('<i>','')
    u = u.replace('</ i>','')
    u = u.replace('</i>','')
    u = u.replace('<i/>','')
    u = u.replace('<i />','')
    return u.strip()
    
