# INTRODUCTION

In this project we trained a Hiearchical Recurrent Neural Network from the paper "A Hiearchical Recurrent Encoder-Decoder for Generative Context-Aware Query Suggestion" by Serban et all, and the paper "A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues", by Sordoni et al. We used the open subtitle corpus found in http://www.opensubtitles.org/.

# DATASET

The open subtitle corpus is commonly used for machine translation, it is composed of movie transcripts between different languages. It has 30 languages with 20,400 files, 150 million tokens and 22 million sentence fragments. We foused on 28 scripts with English subtitles. Originally, we planned to use the CALLHOME corpus (LDC97S42) distributed by the Lingustic Data Constortium. However upon closer examination we noted the overabudance of "hms" from the listener (16% of total utterances) following a long stretch of utterences by the other speaker. Thus we chose the movie corpus, which does not appear to experience this problem.

# PREPROCESSING

A raw example of a movie script is given below:

<pre>
	In the last century before the birth... of the new faith called Christianity... which was destined to overthrow the pagan tyranny of Rome... and bring about a new society... the Roman republic stood at the very centre of the civilized world.

	"Of all things fairest." sang the poet...

	"first among cities and home of the gods is golden Rome." 

	Yet even at the zenith of her pride and power... the Republic lay fatally stricken with a disease called... human slavery.

	The age of the dictator was at hand... waiting in the shadows for the event to bring it forth.

	In that same century... in the conquered Greek province of Thrace... an illiterate slave woman added to her master' s wealth... by giving birth to a son whom she named Spartacus.

	A proud. rebellious son... who was sold to living death in the mines of Libya... before his thirteenth birthday. 

	There. under whip and chain and sun... he lived out his youth and his young manhood... dreaming the death of slavery... 2. 000 years before it finally would die.

	Back to work! 

	Get up, Spartacus, you Thracian dog! 

	Come on, get up!

	My ankle, my ankle!

	My ankle! 

	Spartacus again? 

	This time he dies. 

	Back to work, all of you!
</pre>

Since there are no speakers in this document, 


We used two tokenization schemes. The first is nltk's vanilla tokenization scheme, which includes:

* lowercase all tokens
* removing all non alphanumeric characters.

Note in the last case if a nonalphanumeric character appears inside of a word, then it is removed from the word. For example. Punctuations are not removed.


Next we usedd tworkenize found in tworkenize.py, this include:

* stripping white space
* removing emojis that do not appear consecutively with no space in between
* lower case
* split off edge punctuation

See tworkenize.py for a comprehensive list of tokenization steps. 

Finally, we removed any tweet questions-response pairs where the question is longer than 20 tokens, and the question is shorter than 3 tokens or longer than 20 tokens. 

The vocabulary is limited to 6004 characters, all out of vocabulary (OOV) words are mapped to the token 'unk'.

# MODELS
We used vanilla sequence to sequence model with attention mechanism, following the construction by Vinyals and Kaiser et al. (https://arxiv.org/pdf/1412.7449.pdf). This model was originally designed for machine translation and is trained to maximize the probabilty of target sequence given input sequence, where the cost is cross entropy. The model maps in the input sequence into a hidden vector, where the attention mechanism controls how much hidden information will propogate forward.

We utilized the orginial "translate.py" in the tensorflow repository written for French-English translation. All functions in the original file have been slightly modified for our twitter chatbot. In total, we trained three different models as follow:
	- preprocessed with nltk's vanilla tokenization scheme
	- preprocessed with tworkenize.py
	- preprocessed with tworkenize.py & early stopping
The total vocabulary size for both questions and answers is 6004 including the default special vocabulary used in Seq2Seq model (_PAD_, _UNK_, _GO_, _EOS_). We used four buckets with the following bucket sizes (question, answer)-pairs: [(5, 5), (10, 10), (15, 15), (25, 25)].

# TEST RESULTS and DISCUSSION 
We tested each model with the same set of sentences including some sentences in the training data. The results are in the "results" folder. Detailed explanantions are denoted below.

## 1) Preprocessed with NLTK's Vanilla Tokenization scheme
- learning rate: 0.1
- The total number of training steps: 107600
- Perplexity at the stopping point: 
	- Global: 27.09
	- Buckets: 122.35, 72.53, 70.51, 87.10
- Result files:
	"nltk_107600.txt"
	"nltk_107600.png" (Just screenshot image)

The global perplexity and the buckets' perplexities except the first one decreased as the steps increased, but the first bucket's perplexity decreased at the beginning and increased later. In the results, "unk" appears many times, even for the sentences in the training data. We believe this is due insufficient tokenization. Below is an example from the nltk model.

<pre>
	> Have a great day
	thank you unk
	> It is raining today
	unk on unk
	> Trump got elected as the president of the united states
	unk is the unk unk unk unk
	> It is finally Friday
	its unk unk
	> yeah i'm preparing myself to drop a lot on this man, but definitely need something reliable
	unk unk unk unk unk unk
	> magic's biggest secrets finally revealed on netflix
	unk to get a unk unk
	> just standing up for my friends yo
	unk u unk bro
	> besides if trump say his condolences it won't sound genuine
	so unk so true
	> I am okay with that
	im sorry to unk
	> I love my dad
	fucking fucking unk
	> I love my mom
	whats the unk
	> she is back
	she is unk
	> idk
	unk unk unk
	> idk im just trying to help other people
	i live to unk
	> what's a good place to get free beats?
	its unk to unk
	> let's work
	unk that unk
	> goddamit that degrom image is so good i hope i never lose it
	try here on the unk unk
	> you are so sweet.
	thank you so much
	>
	unk unk unk
	> couldn't help myself
	unk unk unk
	> if i'm being totally honest, i went far out of my way for this one
	i like the same party unk
	> my sister say she ready for me
	lmao yall unk me
</pre>	


## 2) Preprocessed with tworkenize.py
- learning rate: 0.5
- The total number of training steps: 128200
- Perplexity at the stopping point: 
	- Global: 1.11
	- Buckets: 2627.52, 16880.32, 43720.34, 24863.28
- Result files:
	"tworken_128200.txt"
	"tworken_128200.png" 

Note total perplexity score over all buckets dropped to as far as 1.11. However, each bucket's perplextiy kept increasing and eventually became very large. Be that as it may, this model performs qualitatively better than the one trained on the previously tokenized dataset. In particular we observe far less "unk" tokens, and the responses appear more related to the question. 

<pre>
	> Have a great day
	unk your unk
	> It is raining today
	yo unk .
	> Trump got elected as the president of the united states
	unk because i heard . so am always unk of the light .
	> It is finally Friday
	stay ready b
	> yeah i'm preparing myself to drop a lot on this man, but definitely need something reliable
	eh . i saw this unk . good seeing jimmy .
	> magic's biggest secrets finally revealed on netflix
	she's a unk in the red store
	> just standing up for my friends yo
	i can help her way .
	> besides if trump say his condolences it won't sound genuine
	nope . she truly might end his unk .
	> I am okay with that
	the best game .
	> I love my dad
	i cant wait
	> I love my mom
	i love this
	> she is back
	unk is bae
	> idk
	aw .
	>  idk im just trying to help other people
	i'm going attention in the way .
	> what's a good place to get free beats?
	like what the unk works
	> let's work
	work . thanks
	> goddamit that degrom image is so good i hope i never lose it
	what unk help
	> you are so sweet.
	thank you dear !!!!!!
	> couldn't help myself
	the best answer .
	> if i'm being totally honest, i went far out of my way for this one
	you know your unk .
	> my sister say she ready for me
	allowed the type
	> hello
	the last rapper
	> whos the last rapper?
	yeah . unk
	> you mad bro?
	yeah my number
	> I wonder what donald trump did when he was my age
	i thought it was a bitch .
	> i agree with that
	good question .	
</pre>	


## 3) Preprocessed with tworkenize.py & early stopping
- learning rate: 0.5
- The total number of training steps: 27000
- Perplexity at the stopping point:
	- Global: 25.12
	- Buckets: 49.00, 48.75, 79.21, 92.70
- Result files:
	"tworken_earlystopping_27000.txt"
	"tworken_earlystopping_27000.png" (Just screenshot image)

Since perplexity is not the best measure of converation cohesiveness, we hypothesized early stopping may yield better results. The global perplexity at the stopping point of this model is higher than the one of the previous model. All buckets' perplexities of this model are much lower than the ones of the previous model. However, this model appear to generate more 'unk' tokens than the previous two models, and qualitatively perform far worse than the others - most of the answers do not make sense.

<pre>
	> Have a great day
	what unk unk
	> It is raining today
	unk unk unk
	> Trump got elected as the president of the united states
	unk unk unk unk unk unk unk unk unk unk unk unk
	> It is finally Friday
	that's unk unk
	> yeah i'm preparing myself to drop a lot on this man, but definitely need something reliable
	well i was thinking about this one of the unk unk
	> magic's biggest secrets finally revealed on netflix
	that's what i was thinking
	> just standing up for my friends yo
	thanks unk unk
	> besides if trump say his condolences it won't sound genuine
	unk unk unk unk unk unk unk unk unk unk unk
	> I am okay with that
	unk unk unk
	> I love my dad
	you're so unk
	> I love my mom
	my unk unk
	> she is back
	unk unk unk
	> idk
	unk unk unk
	> idk im just trying to help other people
	what i was talking about
	> what's a good place to get free beats?
	i unk unk
	> let's work
	thanks for unk
	> goddamit that degrom image is so good i hope i never lose it
	i can't wait to see unk unk
	> you are so sweet.
	thank you unk !
	> couldn't help myself
	unk unk unk
	> if i'm being totally honest, i went far out of my way for this one
	unk unk unk unk
	> my sister say she ready for me
	lol you know she got it
	> hello
	unk unk unk
	> whos the last rapper?
	unk unk unk
	> you mad bro?
	no right bro
	>  I wonder what donald trump did when he was my age
	i can't believe he was unk
	> i agree with that
	you're right .	
</pre>	

## FUTURE WORK ##

We can improve this model in several ways. First, tokenization appear to make a significant difference in quality, since it changes the distribution over words. Thus it may be worthwhile to consider tokenization schemes that are more appropriate for this domain. Next, we observed that global perplexity is a better measure of performance (qualitatively measured by conversing with the bot) than that of each bucket's. More importantly, constructing a better measure of response quality may yield further improvements. 

It is interesting to note that when the model gives answers with english words, the sentences are grammatically correct. Thus the model appeared to have acquired a "language model". However, this may not be a great use of our data since there is very little of it. Thus we wonder if it is possible for a model to acquire a language model by pretraining on non-conversational english corpus, and then fine tune the parameters on a conversational data. 

Finally, seq-to-seq does not keep a history of previous rounds of conversation. In homework 4, we will extend the model by learning to keep a reprentaton of the history of conversation, and training the model on a dataset where such long term correlation exists. 



