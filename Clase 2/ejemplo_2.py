# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 09:56:45 2022

@author: Acer
"""

import nltk
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

url = 'https://gutenberg.org/cache/epub/2261/pg2261.html'

r = requests.get(url)
html = r.text

# obtenemos el texto puro

from bs4 import BeautifulSoup # procesa y genera el texto puro a partir de ese html

soup = BeautifulSoup(html,'lxml')

soup.head
soup.body
soup.findAll('head')[:2]

text = soup.get_text()

# Paso 1 -> Tokenizar
# Alternativa 1
import re
sentence = "tres tristes tigres comian en tres tristes trigales"
ps = 't\w+' # A-Za-z 0-9 mas de una vez (+)
re.findall(ps, sentence)
tokens = re.findall(r'[A-Z a-z]\w+', text)
tokens[:8]

# Alternativa 2
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer('[A-Z a-z]\w+')
tokens = tokenizer.tokenize(text)
tokens[:8]

# Palabras separadas
import nltk

sw = nltk.corpus.stopwords.words('english')
sw[:25]

# normalizar
lower_tokens = []
for word in tokens:
    lower_tokens.append(word.lower())

words_ns = []
for word in lower_tokens:
    if word not in sw:
        words_ns.append(word)

words_ns[:5]

# Encontrando toda la longitud de las palabras
word_lenghts = [(word, len(word)) for word in words_ns ]
print(f'Lenghts of words: \n {word_lenghts}')

# Creamos una funcion para tomar palabras por ventanas
def sentence_to_ngrams(words_ns, n):
    ngram = []
    for i in range(len(words_ns) - n + 1):
        ngram.append(words_ns[i:i+n])
        #print(ngram)
    return ngram

ngrams = sentence_to_ngrams(words_ns, n = 4)

# Gradicando n-grams
bigrams_series_barh = (pd.Series(nltk.ngrams(words_ns, 2)).value_counts())[:12]
trigrams_series_barh = (pd.Series(nltk.ngrams(words_ns, 3)).value_counts())[:12]

plt.figure(1)
plt.subplot(1,2,1)
bigrams_series_barh.sort_values().plot.barh(color = 'cyan', 
                                            width = 0.8, figsize = (10,8))
plt.title('Bigrams mas frecuentes')
plt.ylabel('Bigrams')
plt.xlabel('Numero de ocurrencia')

plt.subplot(1,2,2)
trigrams_series_barh.sort_values().plot.barh(color = 'cyan', 
                                            width = 0.8, figsize = (10,8))
plt.title('Trigrams mas frecuentes')
plt.ylabel('Trigrams')
plt.xlabel('Numero de ocurrencia')
plt.tight_layout(0.02)


plt.figure(2)
frecuencia = nltk.FreqDist(words_ns)
frecuencia.plot(5)


def plot_word_freq(url):
    r = requests.get(url)
    html = r.text
    soup = BeautifulSoup(html, 'html5lib')
    text = soup.get_text()
    tokenizer = RegexpTokenizer('[A-Z a-z]\w+')
    tokens = tokenizer.tokenize(text)
    sw = nltk.corpus.stopwords.words('english')
    words = [ t.strip().lower() for t in tokens ]
    words_ns = [ word for word in words if word not in sw ]
    
    plt.figure(3)
    freq2 = nltk.FreqDist(words_ns)
    freq2.plot(20)
    
A = len(set(words_ns))