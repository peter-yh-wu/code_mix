# Deep Reranked Code-Switching ASR Model

## Usage

Frist download SEAME datset and put it in tasks/SEAME/data folder. Since SEAME is not a free dataset, you may need to get access to it from [here](https://catalog.ldc.upenn.edu/LDC2015S04).

### Training Language Model

```
cd tasks/lm
python train.py [-h] [--epoch EPOCH] [--model MODEL] [--batch BATCH]
                [--embed_en EMBED_EN] [--embed_cn EMBED_CN] [--hidden HIDDEN]
                [--embed EMBED] [--ngram NGRAM] [--maxlen MAXLEN]
                [--optim OPTIM] [--dp DP] [--mode MODE] [--nworkers NWORKERS]
                [--lr LR] [--mm MM] [--clip CLIP] [--data DATA]
                [--subset SUBSET] [--models_dir MODELS_DIR]
                [--log_dir LOG_DIR] [--gpu_id GPU_ID] [--qg QG]

Language model parameters.

optional arguments:
  -h, --help            show this help message and exit
  --epoch EPOCH         maximum training epochs
  --model MODEL         choose language model
  --batch BATCH         batch size
  --embed_en EMBED_EN   pre-trained word embedding for English
  --embed_cn EMBED_CN   pre-trained word embedding for Chinese
  --hidden HIDDEN       LSTM hidden unit size
  --embed EMBED         word embedding vector size
  --ngram NGRAM         ngram language model
  --maxlen MAXLEN       maximum length of sentence
  --optim OPTIM         optimizer: adadelta, adam or sgd
  --dp DP               dropout rate, float number from 0 to 1.
  --mode MODE           train/test
  --nworkers NWORKERS   number of workers for loading dataset
  --lr LR               initial learning rate
  --mm MM               momentum
  --clip CLIP           gradient clipping
  --data DATA           dataset path
  --subset SUBSET       subset size
  --models_dir MODELS_DIR
                        save model dir
  --log_dir LOG_DIR     logging dir
  --gpu_id GPU_ID       GPU to be used if any
  --qg QG               use QG dataset for data augumentation
```

### Training ASR Model

```
cd tasks/SEAME
cd baseline/
python baseline.py [-h] [--batch-size N] [--save-directory SAVE_DIRECTORY]
                   [--save-all SAVE_ALL] [--epochs N] [--patience PATIENCE]
                   [--num-workers N] [--no-cuda] [--max-data N]
                   [--max-train MAX_TRAIN] [--max-dev MAX_DEV]
                   [--max-test MAX_TEST] [--lr N] [--weight-decay N]
                   [--teacher-force-rate N] [--encoder-dim N]
                   [--decoder-dim N] [--value-dim N] [--key-dim N]
                   [--generator-length N] [--test-mode TEST_MODE]

optional arguments:
  -h, --help            show this help message and exit
  --batch-size N        batch size
  --save-directory SAVE_DIRECTORY
                        output directory
  --save-all SAVE_ALL   saves all epoch models
  --epochs N            number of epochs
  --patience PATIENCE   patience for early stopping
  --num-workers N       number of workers
  --no-cuda             disables CUDA training
  --max-data N          max data in each set
  --max-train MAX_TRAIN
                        max train
  --max-dev MAX_DEV     max dev
  --max-test MAX_TEST   max test
  --lr N                lr
  --weight-decay N      weight decay
  --teacher-force-rate N
                        teacher forcing rate
  --encoder-dim N       hidden dimension
  --decoder-dim N       hidden dimension
  --value-dim N         hidden dimension
  --key-dim N           hidden dimension
  --generator-length N  maximum length to generate
  --test-mode TEST_MODE
                        Test mode: transcript, cer, perp
```

### Beam Search ASR Model Output

```
python test_model.py [-h] [--batch-size N] [--save-directory SAVE_DIRECTORY]
                     [--epochs N] [--patience PATIENCE] [--num-workers N]
                     [--no-cuda] [--max-data N] [--max-train MAX_TRAIN]
                     [--max-dev MAX_DEV] [--max-test MAX_TEST] [--lr N]
                     [--weight-decay N] [--teacher-force-rate N]
                     [--encoder-dim N] [--decoder-dim N] [--value-dim N]
                     [--key-dim N] [--generator-length N]
                     [--test-mode TEST_MODE]
                     [--beam-width [1, 99]]
                     [--lm-path LM_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --batch-size N        batch size
  --save-directory SAVE_DIRECTORY
                        output directory
  --epochs N            number of epochs
  --patience PATIENCE   patience for early stopping
  --num-workers N       number of workers
  --no-cuda             disables CUDA training
  --max-data N          max data in each set
  --max-train MAX_TRAIN
                        max train
  --max-dev MAX_DEV     max dev
  --max-test MAX_TEST   max test
  --lr N                lr
  --weight-decay N      weight decay
  --teacher-force-rate N
                        teacher forcing rate
  --encoder-dim N       hidden dimension
  --decoder-dim N       hidden dimension
  --value-dim N         hidden dimension
  --key-dim N           hidden dimension
  --generator-length N  maximum length to generate
  --test-mode TEST_MODE
                        Test mode: transcript, cer, perp
  --beam-width			Beam search width [1, 99]
  --lm-path LM_PATH     path to pre-trained language model
```


### Reranking Beam Search Results with Language Model

Download our trained language model from [here](https://drive.google.com/open?id=1Vk99nraTk9PnDM7FiQQ7vBh8LDgqYEOp) and put it to tasks/lm/models folder. Then modify the beam search results file path (second argument of `rerank()`, sorry for the inconvenience).

```
python rerank.py
```
