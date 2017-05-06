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

Since there are no speakers in this document, so we first added speakers A and B for each paragraph delimited by the new line symbol. This is certainly a crude approach, but while scanning through the labeled documents it appears to appropriately delimit different speakers most of the time. Next we removed ASCII symbols in an adhoc manner, before using a backported version of tworkenizer created by CMU, which:

	* stripes white space
	* lower case
	* split off edge punctuation.

Note proper nouns and numbers are not folded into on character. The vocabulary is 50,005 characters, this includes 50,000 characters found in corpus, and the special tokens:

	* unk
	* pad
	* go
	* eos - end of sentence
	* eoc - end of conversation

And example of the corpus post processing is as follows:

<pre>
	A: in the last century before the birth of the new faith called christianity which was destined to overthrow the pagan tyranny of rome and bring about a new society the roman republic stood at the very centre of the civilized world
	B: of all things fairest sang the poet
	A: first among cities and home of the gods is golden rome
	B: yet even at the zenith of her pride and power the republic lay fatally stricken with a disease called human slavery

	A: the age of the dictator was at hand waiting in the shadows for the event to bring it forth
	B: in that same century in the conquered greek province of thrace an illiterate slave woman added to her master s wealth by giving birth to a son whom she named spartacus
	A: a proud rebellious son who was sold to living death in the mines of libya before his thirteenth birthday
	B: there under whip and chain and sun he lived out his youth and his young manhood dreaming the death of slavery 2 000 years before it finally would die

	A: back to work
	B: get up spartacus you thracian dog
	A: come on get up
	B: my ankle my ankle
	A: my ankle
</pre>


# MODEL

We used a hiearchical recurrent neural net (HRED) construction found in the paper "A Hierarchical Recurrent Encoder-Decoder for Generative Context-Aware Query Suggestion" by Sordoni, Bengio, Vahabi, Lioma, Simonsen, and Nie. This model assumes the existence of a conversation vector that keeps track of the current state of conversation, and the model will find a representation for this vector, and learn a mapping for this vector across different utterences. Additionally similar to seq-to-seq, HRED learns hidden represenation of words and transition between words within a sentence. Finally, the entire model is trained end-to-end by minimizing entropy of a session, which is composed of multiple rounds of utterances from two speakers.

# TRAINING DETAILS

Since the HRED code base is implemented so that it cannot handle long sessions and long utterances, we limited the conversation to be at most 50 tokens long, and each session has at most four sentences. The sessions were then divided to 20 buckets, the network parameters were:


	* embedding dimension = 64
	* query dimension     = 128
	* session dimension   = 256
	* batch size          = 24

Note the small batch size is used to accommodate the GPU. The network was trained over 72 hours on a GeForce GTX TITAN X gpu.


# TEST RESULTS and DISCUSSION 

The global perplexity and the buckets' perplexities except the first one decreased as the steps increased, but the first bucket's perplexity decreased at the beginning and increased later. In the results, "unk" appears many times, even for the sentences in the training data. We believe this is due insufficient tokenization. Below is an example from the nltk model.

<!-- <pre>
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
 -->

## FUTURE WORK ##

The first concern future works must address is acquiring the right kind of data. If the model assumes there is an underlying dynamic that can be learned in conversation, then this underlying dynamic should exist by inspection. One example of a clear dynamic might be a conversation that transfers from greeting, to well defined turns of question answering, followed by a good bye of sorts. This conversation may appear artificial and, in fact, may need to be constructed in a controlled setting. But this approach has the advantage that the dynamic exists by construction, and we can use the results of the model to determine if it is capable of learning this dynamic, given it exists. On the other hand, it is not clear if this dynamic exists in the movie corpus. In fact it is not clear this dynamic exist in the CALLHOME corpus either. 

Assuming this data set exists, then a prudent course of action might actually to reduce the complexity of the model, in particular the aspect that models the conversation dynamic. We might experiment with a simple autoregressive model (read: no nonlinearity) and benchmark HRED against this model. 


























