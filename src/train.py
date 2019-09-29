#!/usr/bin/env python
# coding: utf-8

# In[26]:


# import dependencies
# %matplotlib inline
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models import LdaModel
from gensim import models, corpora, similarities
import re
from nltk.stem.porter import PorterStemmer
import time
from nltk import FreqDist
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sns.set_style("darkgrid")


# In[28]:


abs_file = sys.argv[1]
text_file = sys.argv[2]


# In[33]:


df_abs = pd.read_csv(abs_file)
df = pd.read_csv(text_file)

# df_abs = pd.read_csv(abs_file)
# df = pd.read_csv(text_file)

df = df.sample(frac=1.0)
df.reset_index(drop=True,inplace=True)
df.head()

df_abs = df_abs.sample(frac=1.0)
df_abs.reset_index(drop=True,inplace=True)
df_abs.head()
temp=0


# In[34]:


def initial_clean(text):
    """
    Function to clean text of websites, email addresess and any punctuation
    We also lower case the text
    """
    text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", text)
    text = re.sub("[^a-zA-Z ]", "", text)
    text = text.lower() # lower case the text
    
    text = nltk.word_tokenize(text)
#     print('sdfdfsfsdf',  text)
    return text

stop_words = stopwords.words('english')
# print(stop_words)
def remove_stop_words(text):
    """
    Function that removes all stopwords from text
    """
#     print(text)
    words = []
    for word in text:
        if word not in stop_words:
            words.append(word)
    return words
#     return [word for word in text if word not in stop_words]

stemmer = PorterStemmer()
def stem_words(text):
    """
    Function to stem words, so plural and singular are treated the same
    """
    try:
#         print('dsfsdfsdfs',text)
#         text = [stemmer.stem(word) for word in text]
#         print(text)
        text = [word for word in text if len(word) > 1] # make sure we have no 1 letter words
    except IndexError: # the word "oed" broke this, so needed try except
        pass
    return text

def apply_all(text):
    """
    This function applies all the functions above into one
    """
    global temp
    temp+=1
    print(temp,flush=True)
    return stem_words(remove_stop_words(initial_clean(text)))


# In[35]:


# clean text and title and create new column "tokenized"
t1 = time.time()
df['tokenized'] = df['paper_text'].apply(apply_all) #+ df['title'].apply(apply_all)
t2 = time.time()
print("Time to clean and tokenize", len(df), "articles:", (t2-t1)/60, "min")
# print(df)


# In[37]:


# first get a list of all words
all_words = [word for item in list(df['tokenized']) for word in item]
# use nltk fdist to get a frequency distribution of all words
fdist = FreqDist(all_words)
# print(all_words)
len(fdist) # number of unique words


# In[38]:


# choose k and visually inspect the bottom 10 words of the top k
# k = 50000
# top_k_words = fdist.most_common(k)
# top_k_words[-10:]


# In[39]:


k = 40000
top_k_words = fdist.most_common(k)
top_k_words[-10:]


# In[40]:


# define a function only to keep words in the top k words
top_k_words,_ = zip(*fdist.most_common(k))
top_k_words = set(top_k_words)
def keep_top_k_words(text):
    return [word for word in text if word in top_k_words]


# In[41]:



# document length
df['doc_len'] = df['tokenized'].apply(lambda x: len(x))
doc_lengths = list(df['doc_len'])
df.drop(labels='doc_len', axis=1, inplace=True)

print("length of list:",len(doc_lengths),
      "\naverage document length", np.average(doc_lengths),
      "\nminimum document length", min(doc_lengths),
      "\nmaximum document length", max(doc_lengths))


# In[44]:


# plot a histogram of document length
num_bins = 1000
fig, ax = plt.subplots(figsize=(12,6));
# the histogram of the data
n, bins, patches = ax.hist(doc_lengths, num_bins, density=1)
ax.set_xlabel('Document Length (tokens)', fontsize=15)
ax.set_ylabel('Normed Frequency', fontsize=15)
ax.grid()
ax.set_xticks(np.logspace(start=np.log10(50),stop=np.log10(2000),num=10, base=10.0))
plt.xlim(0,2000)
ax.plot([np.average(doc_lengths) for i in np.linspace(0.0,0.0035,100)], np.linspace(0.0,0.0035,100), '-',
        label='average doc length')
ax.legend()
ax.grid()
fig.tight_layout()
# plt.show()

# print([np.average(doc_lengths) for i in np.linspace(0.0,0.0035,100)])


# In[70]:


msk = np.random.rand(len(df)) < 0.99
print(sum(msk))


# In[56]:


train_df = df[msk]
train_df.reset_index(drop=True,inplace=True)


# In[57]:


test_df = df[~msk]
test_df.reset_index(drop=True,inplace=True)


# In[58]:


def train_lda(data):
    """
    This function trains the lda model
    We setup parameters like number of topics, the chunksize to use in Hoffman method
    We also do 2 passes of the data since this is a small dataset, so we want the distributions to stabilize
    """
    num_topics = 100
    chunksize = 300
    dictionary = corpora.Dictionary(data['tokenized'])
    corpus = [dictionary.doc2bow(doc) for doc in data['tokenized']]
    t1 = time.time()
    # low alpha means each document is only represented by a small number of topics, and vice versa
    # low eta means each topic is only represented by a small number of words, and vice versa
    lda = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,
                   alpha=1e-2, eta=0.5e-2, chunksize=chunksize, minimum_probability=0.0, passes=2)
    t2 = time.time()
    print("Time to train LDA model on ", len(train_df), "articles: ", (t2-t1)/60, "min")
    return dictionary,corpus,lda


# In[54]:


dictionary,corpus,lda = train_lda(train_df)

from sklearn.externals import joblib

joblib.dump((dictionary, corpus, lda), 'noobhackerz.pkl') 




# # In[59]:


# lda.show_topics(num_topics=10, num_words=20)


# # In[60]:


# lda.show_topic(topicid=4, topn=20)


# # In[61]:


# def jensen_shannon(query, matrix):
#     """
#     This function implements a Jensen-Shannon similarity
#     between the input query (an LDA topic distribution for a document)
#     and the entire corpus of topic distributions.
#     It returns an array of length M where M is the number of documents in the corpus
#     """
#     # lets keep with the p,q notation above
#     p = query[None,:].T # take transpose
#     q = matrix.T # transpose matrix
#     m = 0.5*(p + q)
#     return np.sqrt(0.5*(entropy(p,m) + entropy(q,m)))


# # In[62]:


# def get_most_similar_documents(query,matrix,k=10):
#     """
#     This function implements the Jensen-Shannon distance above
#     and retruns the top k indices of the smallest jensen shannon distances
#     """
#     sims = jensen_shannon(query,matrix) # list of jensen shannon distances
#     return sims
# #     return sims.argsort()[:k] # the top k positional index of the smallest Jensen Shannon distances


# # In[66]:


# doc_topic_dist = np.array([[tup[1] for tup in lst] for lst in lda[corpus]])
# doc_topic_dist.shape


# # In[69]:


# out_val = []
# temp=0
# for random_article_index in range(len(df_abs)):
# #     random_article_index = 1
#     new_bow = dictionary.doc2bow(df_abs.iloc[random_article_index,1])
#     new_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=new_bow)])

#     # this is surprisingly fast
#     most_sim_ids = get_most_similar_documents(new_doc_distribution,doc_topic_dist)
#     out_val.append(most_sim_ids)
# # print(most_sim_ids)
#     temp+=1
#     print(temp)
#     most_similar_df = train_df[train_df.index.isin(most_sim_ids)]

# fl = pd.DataFrame(out_val)
# fl.to_csv("similarity_matrix.csv")


# # In[ ]:




