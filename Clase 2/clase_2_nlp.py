import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

example = "Hoy es un dia increible, ya que vamos a aprender a procesar texto con\
 Python. Es importante conocer nuevos temas de python, y las librerias mas\
 utilizadas en NLP para python para procesar texto"

# Procesamiento con el metodo largo:
words = word_tokenize(example)
print(words)

words = [ w.lower() for w in words if w.isalpha()]
print(words)

stop_words = set(stopwords.words('spanish'))

filtered_sentences = []
for i in words:
  if i not in stop_words:
    filtered_sentences.append(i)

print(filtered_sentences)

# Procedimiento con el metodo corto:
tokens_find = [w.lower() for w in word_tokenize(example) if w.isalpha() ]
print(tokens_find)

filtered_sentences2 = [ w for w in tokens_find if w not in stopwords.words('spanish') ]
print(filtered_sentences)


#Contamos las 3 plabras más frecuentes
from collections import Counter

Counter(filtered_sentences).most_common(3)

#%% Palabraz Raiz

from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

porter_stemmer = PorterStemmer() # Version anterior
snowball_stemmer = SnowballStemmer('spanish') # version nueva

example2 = ['felicidad', 'pensando', 'caminando', 'escuchando', 'bailando', 
            'reina', 'cantante']

for i in example2:
    print(snowball_stemmer.stem(i))
    
example3 = ['happiness', 'thinking', 'pythoner', 'tuning', 'happy', 'food', 
            'dangerous', 'queen']

for i in example3:
    print(porter_stemmer.stem(i))

#%% Etiquetas gramaticales

tokens = nltk.word_tokenize(example)
example4 = set(tokens)

tags = nltk.pos_tag(example4)
print(tags)

nltk.help.upenn_tagset('NN')