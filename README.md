# ull
Repository containing assignments for the Unsupervised Language Learning course, University of Amsterdam.

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
If you add the flag Priors, the KL divergence between $$\mu_w$$ and $$\sigma_w$$ from the Encoder (inputting context and actual word) is compared to $$\mu_s$$ and $$\sigma_s$$ from the prior distribution (candidate).
Else both the actual word and the candidate are given to the encoder and these $$\mu$$ and $$\sigma$$ are compared.


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