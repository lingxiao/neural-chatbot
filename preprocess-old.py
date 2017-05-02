############################################################
# Module  : homework4 - preprocess data  
# Date    : Febuary 28th
# Author  : Xiao Ling  
############################################################

import os
import time
import nltk
import cPickle as pickle
import numpy as np
import itertools
from collections import defaultdict

from utils import *

############################################################
'''
    THE IDs need to conform to this numerical order:

    PAD_ID = 0
    UNK_ID = 1
    GO_ID  = 2
    EOS_ID = 3
    EOQ_ID = 4

    top level routine to preprocess converation:

    @Use: Given:
            - application settings of form (ie):

    RESERVED_TOKENS = {'pad'       : '_'
                      , 'unk'      : '<unk>'
                      , 'go'       : '<go>'
                      , 'eos'      : '</s>'    # end of session
                      , 'eoq'      : '</q>'    # end of query
                      ,'vocab-size': 50005}

            - directory to input raw conversations
            - path to normalized conversations directory 
            - path to normalized index conversations directory 
            - path to normalized concactentation of all index conversations directory 
            - path to w2idx dict
            - path to idx2w dict
            - directory to log 

          * open all files and normalize by:
            - case folding
            - whitespace stripping
            - removing all puncutations
            - removing all meta-information demarked by {...}

          * strip all conversations longer than allowed length

          * construct w2idx and idx2w dictionaries

          * save normalized text, and dictionaries
'''
def preprocessing_convos( RESERVED_TOKENS
                        , input_dir
                        , sess_normed_dir
                        , sess_idx_dir
                        , sess_concat_dir
                        , w2idx_path
                        , idx2w_path
                        , log_dir
                        , max_len    = 100
                        , vocab_size = 50000):

    writer = Writer(log_dir, 1)

    writer.tell('running preprocessing_convos')

    writer.tell('normalizing all sessions ...')

    sess_long = normalize_all( input_dir
                             , RESERVED_TOKENS
                             , writer)

    writer.tell('removing sessions that do not conform to maximum utterance length of ' + str(max_len))

    sessions = []

    rmv = 0

    for sess in sess_long:

        if any(len(xs.split(' ')) > max_len for _,xs in sess):
            writer.tell('removing long session')
            rmv += 1
        else:
            sessions.append(sess)     

    writer.tell('removed ' + str(rmv) + ' sessions')

    '''
        construct tokens for word to index
    '''
    writer.tell ('building idx2w w2idx dictionary ...')

    all_tokens = ' '.join(xs for _,xs in join(sessions))

    idx2w, w2idx, dist  = index(all_tokens, RESERVED_TOKENS, vocab_size)

    '''
        construct sessions encoded according to w2idx
    '''
    writer.tell('constructing encoded sessions ...')

    '''
        make big concactenated version and split
        into 80 percent train and 20 percent validate
    '''
    writer.tell('constructing a version containing the concactenation of all sessions')
    big   = join([xs for _,xs in sess] for sess in sessions)
    cut   = int(float(len(big))*0.8)
    train = big[0:cut]
    test  = big[cut:]

    '''
        construct index version of all sessions
    '''
    writer.tell('encoding all sessions')
    sessions_idx = [[encode(w2idx, RESERVED_TOKENS['unk'], ws) \
                    for _,ws in sess] for sess in sessions] 


    big_idxs   = join(sessions_idx)
    train_idxs = big_idxs[0:cut]
    test_idxs  = big_idxs[cut:]

    '''
        save output
    '''
    writer.tell('saving all nomalized sessions in .txt form ...')

    num = 1

    for sess_norm, sess_idx in zip(sessions, sessions_idx):

        name       = 'sess-' + str(num)
        out_normed = os.path.join(sess_normed_dir, name + '.txt')
        out_idx    = os.path.join(sess_idx_dir   , name + '.npy')

        with open(out_normed, 'w') as h:
            for t,xs in sess_norm:
                h.write(t + ': ' + xs + '\n')

        with open(out_idx, 'wb') as h:
            np.save(h, np.asarray(sess_idx))

        num +=1 

    '''
       save concactenated version of all sessions
       as pkl file
    '''     
    writer.tell('saving all concactenated files')       

    train_path = os.path.join(sess_concat_dir, 'train')
    test_path  = os.path.join(sess_concat_dir, 'test')

    with open(train_path + '.txt','w') as h:
        for xs in train:
            h.write(xs + '\n')

    with open(test_path + '.txt','w') as h:
        for xs in test:
            h.write(xs + '\n')

    with open(train_path + '.pkl','wb') as h:
        pickle.dump(train_idxs,h)

    with open(test_path + '.pkl','wb') as h:
        pickle.dump(test_idxs,h)


    '''
        save w2idx and idx2w
    '''
    writer.tell('saving w2idx and idx2w')

    with open(w2idx_path, 'wb') as h:
        pickle.dump(w2idx, h)

    with open(idx2w_path, 'wb') as h:
        pickle.dump(idx2w, h)
    
    return w2idx, idx2w, sessions


############################################################

'''
    @Use: given w2idx dictionary and unk token, and utterance
          encode into list of indices
    @input: - w2idx :: Dict String Int 
            - unk   :: Stirng
            - ws    :: String
    @Output: indices :: [Int]    
'''
def encode(w2idx, unk, ws):
    words = ws.split(' ')

    bs    = []

    for w in words:
        if w in w2idx: 
            bs.append(w2idx[w])
        else:
            bs.append(w2idx[unk])
    # return np.asarray(bs)
    return bs


'''
    Subrountines for normalizing text

    @Use  : normalize all sessions and
            concat consecutive speaker rounds
    @Input: input_dir  :: String  path to directory holding all
                          raw sessions
            writer     :: Writer
'''
def normalize_all(input_dir, RESERVED_TOKENS, writer):
    '''
        @Input : path/to/file-directory containing list of all phone home transcripts
        @Output: a list of all conversations concactenated together, where
                 each element in list is a tuple of form:
                    (round, sentence) 
                 where each round = 'A' or 'B'

    '''
    writer.tell('Scanning for directory for all files')
    files  = os.listdir(input_dir)
    paths  = [os.path.join(input_dir,f) for f in files]
    convos = [open(p,'r').read()   for p in paths]
    convos = [rs.split('\n') for rs in convos    ]
    convos = [[r for r in rs if r] for rs in convos]

    writer.tell('normalizing every session and concactenate speaker rounds')
    token  = Tokenizer(True, True)
    convos = [normalize_session(s, RESERVED_TOKENS, token) for s in convos]

    return convos



'''
    @Use  : normalize each session and
            concat consecutive speaker rounds

            attatch eos symbol

    @Input: raw session :: [String]
            token       :: Tworkenzier

'''
def normalize_session(sess_raw, RESERVED_TOKENS, Token):

    sess_raw = [xs.split(': ') for xs in  \
               pre_preprocess(sess_raw) \
               if len(xs.split(': ')) == 2]

    sess  = []

    for spkr, utter in sess_raw:

        spkr  = spkr[-1]
        utter = normalize_utterance(Token, utter) 

        if not sess:
            sess.append((spkr, utter))
        else:
            prev_spkr, xs = sess[-1]

            if prev_spkr != spkr:
                sess.append((spkr, utter)) 
            else:
                sess[-1] = (spkr, xs + ' ' + utter)

    #  add end of sentence token
    sess = [(s, utter + ' ' + RESERVED_TOKENS['eos']) for s,utter in sess]

    # add end of conversation
    # s,end_utter = sess[-1]
    # end_utter   = end_utter + ' ' + RESERVED_TOKENS['eoc']
    # return sess[0:-1] + [(s,end_utter)]

    return sess

'''
    @Input: instance of tworkenizer `token`
            a string `rs`
    @Output: normalized string with
                - casefolding
                - whitespace stripping
                - puncutation stripping
                - bespoke normalization specific to this dataset
                    * remove leading punctuations such as: ^, * %
                        * fold all 
'''
def normalize_utterance(token, rs):

    ts = rs.decode('utf-8')
    ts = token.tokenize(ts)
    ys = ' '.join(ts)
    ys = ys.encode('utf-8')
    return ys.strip()


############################################################
'''
    Subroutines for encoding and padding text

    @Use: given a list of tokens and settings with key:
            unk
            pad
            vocab-size
         return word to index

'''
# index :: String 
#       -> Dict String Int 
#       -> (Dict String Int, Dict String Int, nltk.Probability.FreqDist)
def index(tokenized_sentences, tokens, vocab_size):

    tokenized_sentences = [[w] for w in tokenized_sentences.split(' ')]

    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))

    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)

    # index2word
    '''
        reserved tokens must come in this order 
        so the indices align as agreed
    '''
    reserved = [ tokens['pad']
               , tokens['unk']
               , tokens['go'] ]
    
    print reserved               
               # , tokens['eos']
               # , tokens['eoq']]

    '''
        vocab cannot have privilged tokens in it
    '''
    vocab = [(v,n) for v, n in vocab if v not in reserved]

    index2word = reserved + [ x[0] for x in vocab ]

    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)])
    index2word = dict((v,k) for k,v in word2index.iteritems() )

    return index2word, word2index, freq_dist






























