In order to run the "run_seq2seq.py" code, you need "seq2seq_model_rl.py"

> python run_seq2seq.py --data_dir PATH-TO-DATA --train_dir PATH-TO-CHECKPOINTS-FOLDER

For example, in the neural-chatbot folder, 

CUDA_VISIBLE_DEVICES=2 python run_seq2seq.py --data_dir /data2/xiao/neural-inputs/movie/sess-idx --train_dir /data2/xiao/data-neural-chatbot/final/checkpoint --vocab_size 50005

(I made a checkpoints folder in the drl_dialog folder for example.)



