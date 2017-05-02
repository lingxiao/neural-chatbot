############################################################
# Module  : homework4 - preprocess data  
# Date    : Febuary 28th
# Author  : Xiao Ling  
############################################################

import os
import time
import shutil
import cPickle as pickle
import numpy as np


from app import *
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
def normalize_and_index( RESERVED_TOKENS
                        , input_dir
                        , sess_idx_dir
                        , sess_concat_dir
                        , w2idx_path
                        , idx2w_path
                        , max_len    = 100
                        , vocab_size = 50000):

    writer = Writer(config.get_path('log'), 1)
    writer.tell('running normalize_and_index')
    writer.tell('opening preprocessed sessions and normalizing ...')

    sess_long = normalize_all( input_dir
                             , RESERVED_TOKENS
                             , writer)

    writer.tell('found ' + str(len(sess_long)) + ' total sessions')

    writer.tell('removing sessions that do not conform to maximum utterance length of ' + str(max_len))

    sessions = []
    rmv      = 0

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

    w2idx, idx2w  = index(all_tokens, RESERVED_TOKENS, vocab_size)

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
        remove existing files
    '''
    writer.tell('removing existing .npy files')
    shutil.rmtree(sess_idx_dir)
    os.mkdir     (sess_idx_dir)

    '''
        save output
    '''
    writer.tell('saving all ' + str(len(sessions)) + ' normalized sessions in .txt and .npy form ...')

    num = 1

    for sess_norm, sess_idx in zip(sessions, sessions_idx):

        name       = 'sess-' + str(num)
        out_normed = os.path.join(input_dir    , name + '.txt')
        out_idx    = os.path.join(sess_idx_dir , name + '.npy')

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

    writer.tell('Scanning for directory for all files')
    files  = os.listdir(input_dir)
    paths  = [os.path.join(input_dir,f) for f in files]
    Token  = Tokenizer(True, True)

    data = []

    for path in paths:

        with open(path, 'rb') as h:

            session = []

            for line in h:
                if len(line.split(': ')) == 2:
                    spkr, utter = line.split(': ')
                    utter = utter.replace('\n','')
                    utter = normalize_utterance(Token, RESERVED_TOKENS, utter)
                    session.append((spkr, utter))

            data.append(session)


    return data

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
def normalize_utterance(token, RESERVED_TOKENS, rs):

    ts = rs.decode('utf-8')
    ts = token.tokenize(ts)
    ys = ' '.join(ts)
    ys = ys.encode('utf-8')
    return ys.strip() + ' ' + RESERVED_TOKENS['eos']

############################################################
'''
    Subrountines for encoding and padding text

    @Use: given a list of tokens and settings with key:
            unk
            pad
            vocab-size
         return word to index

'''
# index :: String 
#       -> Dict String Int 
#       -> (Dict String Int, Dict String Int)
def index(tokens, RESERVED_TOKENS, vocab_size):

    all_tokens = [x.strip() for x in tokens.split(' ')]

    '''
        construct frequency table of all_tokens
    '''
    freq_dist = { w : 0 for w in set(w for w in all_tokens) }

    for token in all_tokens:
        freq_dist[token] += 1


    freq_dist = [(k,v) for k,v in freq_dist.iteritems()]
    freq_dist = sorted(freq_dist, key=lambda tup: tup[1])
    freq_dist.reverse()

    '''
        reserved tokens must come in this order 
        so the indices align as agreed
    '''
    reserved = [ RESERVED_TOKENS['pad']
               , RESERVED_TOKENS['unk']
               , RESERVED_TOKENS['go'] 
               , RESERVED_TOKENS['eos']
               , RESERVED_TOKENS['eoc']]

    '''
        vocab cannot have these tokens in it either
    '''
    bad_tokens = [ '<i>','@','</i>','/', '<', '>'
                 , '\xc2\xa4', '\xc2\xa4i' ]


    '''
        vocab cannot have privilged tokens or bad tokens
    '''
    vocab = [k for k, _ in freq_dist if k \
            not in reserved + bad_tokens  ]


    vocab = reserved + vocab[0:vocab_size]

    word2index = dict([(w,i) for i,w in enumerate(vocab)])
    index2word = dict((v,k) for k,v in word2index.iteritems() )

    return word2index, index2word





























