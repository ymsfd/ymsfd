+++ 
draft = false
date = 2021-03-12T18:16:59+01:00
title = "Compare text similarity"
description = ""
slug = ""
authors = []
tags = []
categories = []
externalLink = ""
series = []
+++

Textual similarity calculation is an important NLP tasks with a lot of application, for example, search engine, question answering system.
Generally this tasks can be decomposed into three subtasks:
* Text cleaning
* Vector representation
* Distance metrics


# Text Cleaning
Text cleaning typically includes: 
* lower case all words
* reomve special symbols (remove digit)
* remove stopwords (customize stopwords)
* stemming/lemme
* ...

```
import re
import nltk
from nltk.corpus import stopwords

def clean(text):
    lowered = [i.lower() for i in text]
    no_digit = [re.sub(" \d+", " ", tok) for tok in lowered]
    tokens = [nltk.word_tokenize(i) for i in no_digit]
    stop_word = stopwords.words('english')
    exception = ['on', 'off']
    stop_words = [w for w in stop_word if w not in exception]
    stop_words.extend(['?', '!', '/'])
    cleaned = [[w for w in s if w not in stop_words] for s in tokens]
    cleaned = [" ".join(sublist) for sublist in cleaned]
    return cleaned
```


# Vector Representations
Classical text to numerical features transformations are count-based (statistical approach), e.g., Bag-of-Words model and TFIDF. But the statistical
approaches do not take the semantics of words into consideration (synonoms are not closer as they should).
State-of-the-art pretrained language model based text feature representation can grasp the semantical similarity of the text data.

## BoW (baseline)
```
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('filename.tsv', encoding='utf8', sep='\t')
bow = CountVectorizer()
X = bow.fit_transform(df['nl'])
```

## TFIDF
```
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('filename.tsv', encoding='utf8', sep='\t')
tfidf = TfidfVectorizer(analyzer='word')
TV = tfidf.fit_transform(df['nl'])
TV = TV.toarray()
y = df['label']
```

## Fasttext
```
import fasttext

fasttext.FastText.eprint = lambda x: None
fasttext.util.download_model('en', if_exists='ignore')
ft = fasttext.load_model('cc.en.300.bin')
print(len(ft['king']))
print(ft.get_sentence_vector("how are you today"))
```

## Word2Vec
```
from gensim.models.keyedvectors import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, norm_only=True)


def average_sentence(sentence, wv):
    v = np.zeros(300)
    for w in sentence:
        if w in wv:
            v += wv[w]
    return v / len(sentence)

```

## Transformer Encoder

```
from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
sentence_embeddings = model.encode(sentences) # sentences is a list of string sentence (tp list)
print('Sample BERT embedding vector - length', len(sentence_embeddings[0]))
print('Sample BERT embedding vector - note includes negative values', sentence_embeddings[0])
query = "I hate oliver"
query_vec = model.encode([query])[0]


df = pd.read_csv("filename.tsv", sep="\t", names=["col1","col2 "])
col1 = df0["col1"].to_list()
scores = []

for example in col1:
  sim = cosine(query_vec, model.encode([example])[0])
  scores.append(sim)
  
import numpy as np
top5 = np.argsort(scores)[-5:]
top5_match = [tp[i] for i in top5]
```

# Distance Metrics
After the text being vectorized, we can compute the distance/similarity based on the following metrics.

## Cosine Distance
```
def cosine_similarity(u, v):
    sim = 1 - np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return sim
```
```
from gensim.models.keyedvectors import KeyedVectors
# I found out that the n_similarity from gensim library also based on cosine similarity, not sure if result in the same as manually calculated..
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, norm_only=True)
model.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant'])
```
## Jaccard Similarity (X n Y)/(X u Y)
```
def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))
list1 = ['dog', 'cat', 'cat', 'rat']
list2 = ['dog', 'cat', 'mouse']
jaccard_similarity(list1, list2)
``` 

## Levenshtein distance/edit distance
```
import nltk
nltk.edit_distance("humpty", "dumpty")
```

## WMD
``` 
from gensim.models import Word2Vec, KeyedVectors
from gensim.similarities import WmdSimilarity
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


def preprocess(doc):
    doc = doc.lower()
    doc = word_tokenize(doc)
    doc = [w for w in doc if not w in stop_words]
    doc = [w for w in doc if w.isalpha()]
    return doc


text1 = "hot dog"
text2 = "cold cat"
w2v_corpus = [preprocess(text1)]
model = KeyedVectors.load_word2vec_format('', binary=True, limit=5000)
model.init_sims(replace=True)
num_best = 1
instance = WmdSimilarity(w2v_corpus, model, num_best=num_best)
query = [preprocess(text2)]
sims = instance[query]
similarity = sims[0][1]
print("the sentence are ", similarity, "% similar")
```
