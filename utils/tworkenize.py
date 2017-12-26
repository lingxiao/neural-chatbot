# -*- coding: utf-8 -*-
"""Twitter and web aware tokenizer for English.

This is a backport of CMU's Twoknizes, which has a colorful history:

1) Brendan O'Connor wrote original version in Python:

       http://github.com/brendano/tweetmotif

2a) Kevin Gimpel and Daniel Mills modified it for POS tagging for the CMU ARK
    Twitter POS Tagger
2b) Jason Baldridge and David Snyder ported it to Scala
3) Brendan bugfixed the Scala port and merged with POS-specific changes for
   the CMU ARK Twitter POS Tagger
4) Tobi Owoputi ported it back to Java and added many improvements
5) I backported it from Java to Python for integration with Python-based
   word embedding training code that needed to be used on a large-scale
   collection of discussion forum and newswire data.

References
----------
- http://github.com/brendano/tweetmotif
- https://github.com/brendano/ark-tweet-nlp/blob/master/src/cmu/arktweetnlp/Twokenize.java
"""
from __future__ import unicode_literals

import regex as re


__all__ = ['Tokenizer']


RE_FLAGS = re.VERSION0 | re.UNICODE


#########################################
# Convenience functions
#########################################
def regex_or(*items):
    r = '|'.join(items)
    r = '(?:' + r + ')'
    return r


def pos_lookahead(r):
    return '(?=' + r + ')'


def neg_lookahead(r):
    return '(?!' + r + ')'


def pos_lookbehind(r):
    return '(?<=' + r + ')'


def neg_lookbehind(r):
    return '(?<!' + r + ')'


def optional(r):
    return '(%s)?' % r


#########################################
# REGEX PATTERNS FOR PROTECTED SEQUENCES
#########################################
# Punctuation.
PUNCT_CHARS = '''['"“”‘’.。¿?¡!…,:;]'''
PUNCT_SEQ = '''['"“”‘’]+|[.。¿?¡!,…]+|[:;]+'''

# URLS.
ENTITY = '&(?:amp|lt|gt|quot);'
URL_START1 = regex_or('https?://', r'\bwww\.')
COMMON_TLDS = regex_or('com', r'co\.uk', 'org', 'edu', 'gov', 'net', 'mil',
                       'aero', 'asia', 'biz', 'cat', 'coop', 'info', 'int',
                       'jobs', 'mobi', 'museum', 'name', 'pro', 'tel',
                       'travel', 'xxx', 'ca')
COMMON_CC_TLDS = regex_or('ac', 'ad', 'ae', 'af', 'ag', 'ai', 'al', 'am', 'an', 'ao',
                          'aq', 'ar', 'as', 'at', 'au', 'aw', 'ax', 'az', 'ba', 'bb',
                          'bd', 'be', 'bf', 'bg', 'bh', 'bi', 'bj', 'bm', 'bn', 'bo',
                          'br', 'bs', 'bt', 'bv', 'bw', 'by', 'bz', 'ca', 'cc', 'cd',
                          'cf', 'cg', 'ch', 'ci', 'ck', 'cl', 'cm', 'cn', 'co', 'cr',
                          'cs', 'cu', 'cv', 'cx', 'cy', 'cz', 'dd', 'de', 'dj', 'dk',
                          'dm', 'do', 'dz', 'ec', 'ee', 'eg', 'eh', 'er', 'es', 'et',
                          'eu', 'fi', 'fj', 'fk', 'fm', 'fo', 'fr', 'ga', 'gb', 'gd',
                          'ge', 'gf', 'gg', 'gh', 'gi', 'gl', 'gm', 'gn', 'gp', 'gq',
                          'gr', 'gs', 'gt', 'gu', 'gw', 'gy', 'hk', 'hm', 'hn', 'hr',
                          'ht', 'hu', 'id', 'ie', 'il', 'im', 'in', 'io', 'iq', 'ir',
                          'is', 'it', 'je', 'jm', 'jo', 'jp', 'ke', 'kg', 'kh', 'ki',
                          'km', 'kn', 'kp', 'kr', 'kw', 'ky', 'kz', 'la', 'lb', 'lc',
                          'li', 'lk', 'lr', 'ls', 'lt', 'lu', 'lv', 'ly', 'ma', 'mc',
                          'md', 'me', 'mg', 'mh', 'mk', 'ml', 'mm', 'mn', 'mo', 'mp',
                          'mq', 'mr', 'ms', 'mt', 'mu', 'mv', 'mw', 'mx', 'my', 'mz',
                          'na', 'nc', 'ne', 'nf', 'ng', 'ni', 'nl', 'no', 'np', 'nr',
                          'nu', 'nz', 'om', 'pa', 'pe', 'pf', 'pg', 'ph', 'pk', 'pl',
                          'pm', 'pn', 'pr', 'ps', 'pt', 'pw', 'py', 'qa', 're', 'ro',
                          'rs', 'ru', 'rw', 'sa', 'sb', 'sc', 'sd', 'se', 'sg', 'sh',
                          'si', 'sj', 'sk', 'sl', 'sm', 'sn', 'so', 'sr', 'ss', 'st',
                          'su', 'sv', 'sy', 'sz', 'tc', 'td', 'tf', 'tg', 'th', 'tj',
                          'tk', 'tl', 'tm', 'tn', 'to', 'tp', 'tr', 'tt', 'tv', 'tw',
                          'tz', 'ua', 'ug', 'uk', 'us', 'uy', 'uz', 'va', 'vc', 've',
                          'vg', 'vi', 'vn', 'vu', 'wf', 'ws', 'ye', 'yt', 'za', 'zm',
                          'zw')
URL_START2 = (r'\b(?:[A-Za-z\d-])+' + # Sequences of characters/digits.
             r'(?:\.[A-Za-z\d]+){0,3}\.' +  # 0-3 additional such groups.
             regex_or(COMMON_TLDS, COMMON_CC_TLDS) + # Top-level domains.
             r'(?:\.' + COMMON_CC_TLDS + ')?' +          # Optional second top-level domains.
             pos_lookahead(r'\W|$'))
URL_BODY = r'(?:[^\.\s<>][^\s<>]*?)?' # Fails on https://www.cia.gov/library/publicat.../2187rank.html and the like.
URL_EXTRA_CRAP_BEFORE_END = '%s+?' % regex_or(PUNCT_CHARS, ENTITY)
URL_END = regex_or( r'\.\.+', r'[<>]', r'\s', '$')
URL = (regex_or(URL_START1, URL_START2) +
      URL_BODY +
      pos_lookahead( optional(URL_EXTRA_CRAP_BEFORE_END) + URL_END))

# Email addresses.
BOUND = '(?:\\W|^|$)'
EMAIL = ('(?<=' + BOUND +
         r')[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}(?=' + BOUND + ')')


# Twitter parlance.
HASHTAG = '#[a-zA-Z0-9_]+'
AT_MENTION = '[@＠][a-zA-Z0-9_]+'


# Emoticons.
NORMAL_EYES = r'[:=]'
WINK = r'[;]'
NOSE_AREA = r'(?:|-|[^a-zA-Z0-9 ])'
HAPPY_MOUTHS = r'[D\)\]\}]'
SAD_MOUTHS = r'[\(\[\{]'
TONGUE = r'[pPd3]+'
OTHER_MOUTHS = r'(?:[oO]+|[/\\]+|[vV]+|[Ss]+|[|]+)'
BF_LEFT = r"(♥|0|o|°|v|\$|t|x|;|\u0CA0|@|ʘ|•|・|◕|\^|¬|\*)"
BF_CENTER = r"(?:[\.]|[_-]+)"
BF_RIGHT = r"\2"
S3 = r"(?:--['\"])"
S4 = r"(?:<|&lt;|>|&gt;)[\._-]+(?:<|&lt;|>|&gt;)"
S5 = "(?:[.][_]+[.])"
BASIC_FACE = ("(?:(?i)" + BF_LEFT + BF_CENTER + BF_RIGHT + ")|" + S3 + "|" +
              S4 + "|" + S5)
EE_LEFT = "[＼\\\\ƪԄ\\(（<>;ヽ\\-=~\\*]+"
EE_RIGHT= r"[\-=\);'\u0022<>ʃ）/／ノﾉ丿╯σっµ~\\*]+"
EE_SYMBOL = r"[^A-Za-z0-9\s\(\)\*:=-]"
EAST_EMOTE = EE_LEFT + "(?:" + BASIC_FACE + "|" + EE_SYMBOL +")+" + EE_RIGHT

EMOTICON = (regex_or(NORMAL_EYES, WINK) +
            NOSE_AREA +
            regex_or(TONGUE, OTHER_MOUTHS, SAD_MOUTHS, HAPPY_MOUTHS))

# Various symbols.
HEARTS = '(?:<+/?3+)+' # Additional hearts listed in DECORATIONS.
ARROWS = '(?:<*[-―—=]*>+|<+[-―—=]*>*)'
DECORATIONS = regex_or('[♫♪]+', u'[★☆]+', '[♥❤♡]+', r'[\u2639-\u263b]+',
                       r'[\ue001-\uebbb]+')

# Abbreviations.
def regexify_abbrev(abbrev):
    icase = []
    for c in abbrev:
        if c == '.':
            icase.append(r'\.')
        else:
            icase.append(r"[%s%s]" % (c, c.upper()))
    pattern = r'\b%s' % ''.join(icase)
    return pattern


TITLES = ['mr.', 'messrs.',
          'mrs.', 'mmes.',
          'ms.',
          'dr.', 'drs.',
          'prof.', 'rev.', 'hon.', 'st.',
          'sr.', 'jr.',
          'ph.d.', 'm.d.','b.a.', 'm.a.', 'd.d.s',
           'gen.', 'rep.', 'sem.']
STREETS = ['st.', 'dr.', 'ave.', 'blvd.', 'cir.', 'crt.', 'ct.']
OTHER_ABBREVS = ['a.m.', 'p.m.', 'u.s.', 'u.s.a.',
                 'i.e.', 'e.g.',
                 'a.d.', 'c.e.', 'b.c.', 'b.c.e.',
                 'd.c.']
ABBREVS1  = TITLES + STREETS + OTHER_ABBREVS
ABBREVS   = regex_or(*[regexify_abbrev(abbrev) for abbrev in ABBREVS1])

BOUNDARY_NOT_DOT = regex_or('$', r'\s', r'[“"?!,:;]', ENTITY)
AA1       = r'(?:[A-Za-z]\.){2,}' + pos_lookahead(BOUNDARY_NOT_DOT)
AA2       = r'(?:[A-Za-z]\.){1,}[A-Za-z]' + pos_lookahead(BOUNDARY_NOT_DOT)

ARBITRARY_ABBREV = regex_or(AA1, AA2)


# Numbers.
TIMELIKE        = r'\d+(?::\d+){1,2}'
NUM_NUM         = r'\d+\.\d+'
NUM_WITH_COMMAS = (r'(?:(?<!\d)\d{1,3},)+?\d{3}' +
                   pos_lookahead(regex_or('[^,\d]','$')))
CURRENCY        = regex_or(u'[$£¥ƒ]', r'[\u20A0-\u20CF]')
NUM_COMB        = CURRENCY + r"?\d+(?:\.\d+)+%?"

# Miscellaneous other.
SEPARATORS = regex_or('--+', '―', '—', '~', '–', '=')
THINGS_THAT_SPLIT_WORDS = '[^\\s\\.,?\"]'
EMBEDDED_APOSTROPHE = (THINGS_THAT_SPLIT_WORDS + "+['’′]" +
                       THINGS_THAT_SPLIT_WORDS  +  '*')

# Combine into one BIG pattern.
PROTECT = [HEARTS,
           URL,
           EMAIL,
           TIMELIKE,
           NUM_NUM,
           NUM_WITH_COMMAS,
           NUM_COMB,
           EMOTICON,
           ARROWS,
           ENTITY,
           PUNCT_SEQ,
           ABBREVS,
           ARBITRARY_ABBREV,
           SEPARATORS,
           DECORATIONS,
           EMBEDDED_APOSTROPHE,
           HASHTAG,
           AT_MENTION,
           ]
PROTECT_PATTERN = regex_or(*PROTECT)


#########################################
# Additional patterns
#########################################

WHITESPACE       = r'\s+'

EDGE_PUNCT       = r"""['"\p{Pi}\p{Pf}\p{Ps}\p{Pe}\p{So}]"""
NOT_EDGE_PUNCT   = r"""[a-zA-Z0-9]""" # Content characters.
OFF_EDGE         = r"(^|$|:|;|\s|\.|,)" # Colon/semicolon also belong to EDGE_PUNCT
                                #  (via \p{Po}).
EDGE_PUNCT_LEFT  = '%s(%s+)(%s)' % (OFF_EDGE, EDGE_PUNCT, NOT_EDGE_PUNCT)
EDGE_PUNCT_RIGHT = '(%s)(%s+)%s' % (NOT_EDGE_PUNCT, EDGE_PUNCT, OFF_EDGE)

MORE_PUNCT = '[\-+―—~–=|_^]+'
ALLPUNCT_SEQ = '^%s$' % regex_or(PUNCT_SEQ, EDGE_PUNCT, MORE_PUNCT)

#########################################
# Actual tokenizer.
#########################################
class Tokenizer(object):
    """Twitter and web aware English tokenizer.

    This is a backport of CMU's Twokenize from Java to Python with some
    minor modifications required for my own research.

    References
    ----------
    https://github.com/brendano/ark-tweet-nlp/blob/master/src/cmu/arktweetnlp/Twokenize.java

    Parameters
    ----------
    casefold : bool, optional
        If True text is lowercased prior to tokenization.
        (Default: False)

    elim_punct : bool, optional
        If True sequences of punctuation symbols are omitted from the returned
        tokens.
        (Default: False)

    Attributes
    ----------
    whitespace_reo
    allpunct_seq_reo
    left_edge_punct_reo
    right_edge_punct_reo
    protected_reo
    """
    whitespace_reo = re.compile(WHITESPACE, RE_FLAGS)
    allpunct_seq_reo = re.compile(ALLPUNCT_SEQ, RE_FLAGS)
    left_edge_punct_reo = re.compile(EDGE_PUNCT_LEFT, RE_FLAGS)
    right_edge_punct_reo = re.compile(EDGE_PUNCT_RIGHT, RE_FLAGS)
    protected_reo = re.compile(PROTECT_PATTERN, RE_FLAGS)

    def __init__(self, casefold=False, elim_punct=False):
        self.__dict__.update(locals())
        del self.self

    def _postprocess(self, tokens):
        """Postprocess tokenization to remove punctuation and, optionally,
        punctuation.

        Parameters
        ----------
        tokens : list of str
            Tokenization.

        Returns
        -------
        new_tokens : list of str
            Postprocessed version of ``tokens``.
        """
        new_tokens = []
        for token in tokens:
            # Be extra safe in case our regexes went awry.
            if token == '' or token == []:
                continue

            # Eliminate punctuation.
            if self.elim_punct:
                if not self.allpunct_seq_reo.match(token) is None:
                    continue

            new_tokens.append(token)

        return new_tokens

    def tokenize(self, text):
        """Tokenize text.

        Parameters
        ----------
        text : str
            Text to be tokenized.

        Returns
        -------
        tokens : list of str
            Tokenization.
        """
        # Optional casefolding.
        if self.casefold:
            text = text.lower()

        # Squeeze off whitespace.
        text = self.whitespace_reo.sub(' ', text)

        # Split off edge punctuation.
        text = self.left_edge_punct_reo.sub(r'\1\2 \3', text)
        text = self.right_edge_punct_reo.sub(r'\1 \2\3', text)

        # Identify all portions of string to leave as is (protected) and
        # tokenize JUST the unprotected sequences.
        protected = []
        unprotected = []
        ii = 0
        for mo in self.protected_reo.finditer(text):
            unprotected.append( (ii, mo.start()) )
            protected.append( mo.span() )
            ii = mo.end()
        unprotected.append( (ii, len(text)) )
        protected = [text[bi:ei] for bi, ei  in protected]
        unprotected = [text[bi:ei] for bi, ei  in unprotected]
        unprotected = [s.split() for s in unprotected]

        # Combine the tokenizations of protected and unprotected spans.
        tokens = []
        ii = 0
        for ii, protected_token in enumerate(protected):
            tokens.extend(unprotected[ii])
            tokens.append(protected_token)
        tokens.extend(unprotected[-1])

        # Postprocess tokenization to remove empty tokenizations and,
        # optionally, punctuation.
        tokens = self._postprocess(tokens)

        return tokens
















        