import nltk

from gensim.models import Word2Vec
from nltk.corpus import stopwords

import re

paragraph = """Cartoons have been a cherished part of our childhood memories, leaving us with lasting impressions and endless joy. Growing up, we were captivated by a world where heroes and villains, friendships and adventures, were all brought to life in vibrant colors. From Tom and Jerry's timeless chase to Scooby-Doo's thrilling mysteries, these characters were more than mere entertainment; they were our first glimpse into humor, imagination, and camaraderie. These stories taught us lessons on bravery, resilience, and kindness, all while keeping us engaged with laughter and suspense. Despite the advancement in technology and changing times, the classic cartoons from our past hold a special place in our hearts, reminding us of a simpler time when life’s biggest problems could be solved with friendship, laughter, and a bit of creativity. Our favorite cartoons were not just stories on a screen; they were companions, shaping our imaginations and becoming a part of who we are today."""



# text Preprocessing the data
text = re.sub(r'\[[0-9]*\]',' ',paragraph)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)

# Preparing the dataset
sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]
    
    
# Training the Word2Vec model
model = Word2Vec(sentences, min_count=1)
#if the word is present < then 1 then use to skip the  conunt and as my data is very small 
#word2vec is applied for huge amount of data

words = model.wv.vocab
# in this paragrapb if we want to find the vocalbulary & create a object called words
# if you select then each & every word there may be vectors and dimensions associated to it
# 


# Finding Word Vectors
vector = model.wv['childhood']
#if i want to find the vector of war word and if i want to find the relationship 

# Most similar words
similar = model.wv.most_similar('childhood')
#if i try to find most similar word related to the war 

similar = model.wv.most_similar('memories')

similar = model.wv.most_similar('favorite')



#STILL MORE RESEARCH GOING ON REGARDS TO THE WORD2VEC
