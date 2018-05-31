# ull
Repository containing assignments for the Unsupervised Language Learning course, University of Amsterdam.

# Instructions for Lab 3:

For this lab the SentEval library is needed: ```https://github.com/facebookresearch/SentEval```.
We evaluate three models that learn word embeddings: Skip-Gram Negative Sampling, Bayesian Skip-Gram and Embed-Align.
We combine these models with three methods for computing sentence representations: summing and normalising (AVG), weighted summing (SIF) and a recurrent encoder (GRAN).
For our results, see the table pdf in the Lab3 folder.

### SentEval

We used two scripts to evaluate our models using SentEval:

```senteval_custom.py```, using pickled embeddings that are used in SentEval. It can be run in three ways:
1. ```python3 senteval_custom.py rnn embeddings.pickle encoder.pt w2i.pickle```
2. ```python3 senteval_custom.py avg embeddings.pickle```
3. ```python3 senteval_custom.py sif embeddings.pickle corpus.en FLOAT``` Where the float is a smoothing parameter for the smoothed inverse frequency weighting scheme.

```senteval_embedalign.py```, that can simply be run by adding ```python3``` in front. This uses a pretrained Embed-Align model.

### Learn Skip-Gram

To train Skip-Gram Negative Sampling embeddings with Gensim, run ```python3 learn_sgns.py```.

### Learn a Bayesian Skip-Gram model

For this we refer to the instructions for the second lab below.

### Learn a Gated Recurrent Averaging Network (GRAN)

For this we refer to the folder SentEmbed, containing all code to train a GRAN encoder.
To train the encoder, run the following command:

```
 python3 train.py --batch_size INT --epochs INT --enc_type gran --lr FLOAT --tf_ratio FLOAT --embed embeddings.pickle [--enable_cuda]
```
After training, an encoder and a dictionary mapping words to indices are saved. Use these for the ```senteval_custom.py``` script.

# Instructions for Lab 2:

For this lab we implemented three models: Skipgram Negative Sampling, Bayesian Skip-Gram and Embed-Align. In the _Lab2_ folder you will find separate folders for each of them. The models trained were too large to include, but we did include the embeddings in a pickled file. If you load them they can be evaluated with the testing script of Skipgram Negative Sampling. They are pickled dictionaries mapping words to arrays.

### Skipgram Negative Sampling

Run the following command to train a model:
```
python3 train.py --corpus TEXTFILE --batch_size INT --lr FLOAT --dim INT --epochs INT --nr_sents INT --window INT --neg_samples INT --valid_sentences LSTSENTENCES --valid_candidates LSTCANDIDATES [--enable_cuda]
```
Arguments needing a little bit more explanation: 
 - corpus: Text file with one sentence per line.
 - nr_sents: How many lines to use from the corpus. By setting it to -1 all will be used.

Test a model:
 - Either a _.pt_ model: ```python3 test.py --embed_file MODEL.pt --model --window INT --test_sentences LSTSENTENCES --test_candidates LSTCANDIDATES [--both] [--multiply]```.
 - Or just pickled embeddings: ```python3 test.py --embed_file MODEL.pickle --window INT --test_sentences LSTSENTENCES --test_candidates LSTCANDIDATES [--multiply]```.
When adding the flag ```--both``` the input and output embeddings from SGNS will be used. This is only possible when you input an entire _.pt_ model.
When adding the falg ```--multiply``` a sentence representation is created by multiplying vectors instead of adding them.

### Bayesian Skip-Gram

Run the following command to train a model:
```
python3 train.py --corpus TEXTFILE --nr_sents INT --dim INT --lr FLOAT --window INT --valid LSTSENTENCES --candidates LSTCANDIDATES --gold LSTGOLD [--enable_cuda]
```

Test a model:

```
python3 test.py --encoder MODEL.PT --decoder MODEL.PT --test_sentences LSTSENTENCES --test_candidates LSTCANDIDATES [--priors]
```
If you add the flag Priors, the KL divergence between mu_w and sigma_w from the Encoder (inputting context and actual word) is compared to mu_s and sigma_s from the prior distribution (candidate).
Else both the actual word and the candidate are given to the encoder and these mu and sigma are compared.


### Embed Align

Run the following command to train a model:
```
python3 train.py --corpus TEXTFILE --nr_sents INT --unique_words INT --dim INT --lr FLOAT --window INT --valid LSTSENTENCES --candidates LSTCANDIDATES --gold LSTGOLD [--enable_cuda]
```

Test a model:

```
python3 test.py --encoder MODEL.PT --decoder MODEL.PT --test_sentences LSTSENTENCES --test_candidates LSTCANDIDATES
```

# Instructions for Lab 1:

The code for lab 1 can be found in the Lab1 folder. We separated functionality for the three main topics into three files:
1. ULL_Lab1_WordSimilarity.ipynb
2. ULL_Lab1_WordAnalogy.ipynb
3. ULL_Lab1_Clustering.ipybn

You can run the code by first setting the correct local paths to the data in the first code cell, and then run the code cell by cell.
The requirements are listed in the first cell of every notebook.
For the first two files, the tables and figures are shown in markdown cells. If you want to reproduce them, set the flag create_figures to True.
Note that especially the analogy task will take a long time to finish (~20 minutes).