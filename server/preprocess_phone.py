############################################################
# Module  : preprocess phone data  
# Date    : Febuary 28th
# Author  : Xiao Ling  
############################################################

import os
import shutil
from app import *
from utils import *

############################################################
'''
    @Use: Given:

            - directory to input raw phone conversations
            - path to normalized conversations directory 

          * open all files and normalize by:
            - removing all meta-information demarked by {...}
            - fold speaker turns

'''
def preprocess_phone(input_dir = '', out_dir = ''):

    writer = Writer(config.get_path('log'), 1)

    writer.tell('running preprocessing phone ...')

    writer.tell('normalizing all sessions ...')

    sessions = preprocess_all(input_dir, writer)

    '''
        remove existing files
    '''
    writer.tell('removing existing preprocessed files')
    shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    '''
        save output
    '''
    writer.tell('saving all preprocessed sessions in .txt form ...')

    num = 1

    for sess in sessions:

        name    = 'sess-' + str(num)
        out_pre = os.path.join(out_dir, name + '.txt')

        with open(out_pre, 'w') as h:
            for t,xs in sess:
                h.write(t + ': ' + xs + '\n')

        num +=1 

############################################################

'''
    Subrountines for normalizing text

    @Use  : normalize all sessions and
            concat consecutive speaker rounds
    @Input: input_dir  :: String  path to directory holding all
                          raw sessions
            writer     :: Writer
'''
def preprocess_all(input_dir, writer):
    '''
        @Input : path/to/file-directory containing list of all phone home transcripts
        @Output: a list of all conversations concactenated together, where
                 each element in list is a tuple of form:
                    (round, sentence) 
                 where each round = 'A' or 'B'

    '''
    writer.tell('Scanning for directory for all files')
    files  = os.listdir(input_dir)
    paths  = [os.path.join(input_dir,f) for f in files if '.txt' in f]
    convos = [open(p,'r').read()   for p in paths]
    convos = [rs.split('\n') for rs in convos    ]
    convos = [[r for r in rs if r] for rs in convos]

    writer.tell('normalizing every session and concactenate speaker rounds')

    convos = [preprocess_session(s) for s in convos]

    return convos

'''
    @Use  : preprocess each session and
            concat consecutive speaker rounds

            attatch eos symbol

    @Input: raw session :: [String]

'''
def preprocess_session(sess_raw):

    sess_raw = [xs.split(': ') for xs in  \
               pre_preprocess(sess_raw) \
               if len(xs.split(': ')) == 2]

    sess  = []

    for spkr, utter in sess_raw:

        spkr  = spkr[-1]

        if not sess:
            sess.append((spkr, utter))
        else:
            prev_spkr, xs = sess[-1]

            if prev_spkr != spkr:
                sess.append((spkr, utter)) 
            else:
                sess[-1] = (spkr, xs + ' ' + utter)

    return sess

def pre_preprocess(convo):
    return [' '.join(fold_gesture(strip_word_punc(t)) \
           for t in cs.split()) for cs in convo]

def strip_word_punc(token):
    '''
        @Input : one word token
        @Output: maps all these:
                    ^tok, *tok, %tok, ~tok
                    ((tok))
                to tok

    '''
    if not token:
        return token

    else:
        token = token.strip()
        to = token[0]
        tl = token[-1]

        if to in ['^','*','%', '~', '{','@', '+']:
            return strip_word_punc(token[1:])

        elif tl in ['-', '}']:
            return strip_word_punc(token[0:-1])

        elif token[0:2] == '((':
            return strip_word_punc(token[2:])

        elif token[-2:] == '))':
            return strip_word_punc(token[0:-2])

        elif to == '<' and tl == '>':
            return strip_word_punc(token[1:-1])
        else:
            return token

def fold_gesture(token):
    '''
        @Input : one word token
        @Output: maps all these:
                    {tok} 
                    [tok]
                to emtpy string
    '''
    if not token: return token

    else:
        to  = token[0]
        tl  = token[-1]
        tok = ''

        if to == '{' and tl == '}' \
        or to == '[' and tl == ']' \
        or token == '(( ))':
            return tok
        elif token == '((' \
        or   token == '))':
            return ''
        else:
            return token



