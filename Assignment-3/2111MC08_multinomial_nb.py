'''
Title : Multinomial Naive bayes for given documents
Name : Omkar Santosh Gavhane
Roll No : 2111MC08
Email : omkar_2111mc08@iitp.ac.in
'''

import csv
import re
import nltk
import numpy as np
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#Given documents
train=["In recent years, researchers, for computer vision, have proposed many deep learning (DL) methods for various tasks, and facial recognition (FR) made an enormous leap using these techniques",
"Deep FR systems benefit from the hierarchical architecture of the deep learning methods to learn discriminative face representation",
"Computer vision methods have been widely used for facial recognition"]
test=["Deep learning based computer vision methods have been used for facial recognition"]
#Class-DL=>1,Class-CV=>0
class_dict={0:"CV",1:"DL"}
Class=[1,1,0]

def preprocess_doc(data):
    '''Here we are iterating through each document.
    and from each document only the word containing letter a to z and A to Z
    are cosidered then convert it into lower case and then lemmatize word only if it
    is not in stopwords and then create a new document and then list of documents are returned 
    along with vocabulary'''
    #lemmatization
    lemmatizer=WordNetLemmatizer()
    vocab=[]
    new_d=[]
    for e in data:
        di=re.sub('[^a-zA-Z]',' ',e)
        di=di.lower()
        di=di.split()
        di=[lemmatizer.lemmatize(word) for word in di if word not in stopwords.words('english')]
        new_d.append(di)
        vocab.extend(di)
    #selecting only unique words
    vocab=list(set(vocab))
    return new_d,vocab

def prob_word_given_class(vectors,word_cno,target):
    '''Here we want probability of word given target class.
    as target class of doc is present as last element of vector
    representation of each doc in vectors i.e vectors=[[1,0,0,1,0,...,1],[1,0,1,0,...,0],...]]
    here doc1 is represented as [1,0,0,1,0,...,1] and last element is target class'''
    #now we are selecting all  docs whose last element is given target
    target_docs=[e for e in vectors if e[-1]==target]
    '''to count the no of docs with given words in given target class we are
    performing transpose operation on target_docs and then counting the no of
    docs with given word'''
    transpose_target_docs=np.array(target_docs).T.tolist()
    no_docs_with_word=sum(transpose_target_docs[word_cno])
    #finally returning the p(word/target)
    return (no_docs_with_word+1)/(len(target_docs)+len(vectors[0]))

def prob_words_given_class(vocab,vectors):
    '''it calculates the probability of word given target class for all word and target classes
    and maintaines that value in dictionary as
        words_prob={'word1':[p0,p1],'word2':[p0,p1],...,'wordn':[p0,p1]}
    where
        p0=probability of wordi given target class is 0 and
        p1=probability of wordi given target class is 1
    it calculates the probability of wordi given target class using
    prob_word_given_class() function'''
    words_prob={}
    for i in range(len(vocab)):
        words_prob[vocab[i]]=[]
        words_prob[vocab[i]].append(prob_word_given_class(vectors,i,0))
        words_prob[vocab[i]].append(prob_word_given_class(vectors,i,1))
    return words_prob

def vectorize(train_d,train_vocab):
    '''After creating vocabulary we need to create a representation for document
    here for each document we are iterating over all vocabulary words 
    and then checking whether word from vocabulary is in document 
    if yes then set the position as 1 else 0 in vector whose size is equal to size of vocabulary'''
    vectors=[]
    cnt=0
    for di in train_d:
        v=[]
        for i in range(len(train_vocab)):
            v.append(di.count(train_vocab[i]))
        v.append(Class[cnt])
        cnt+=1
        vectors.append(v)
    return vectors

def nb_multinomial():
    #preprocessing the training data
    train_d,train_vocab=preprocess_doc(train)
    #vectorize the training data
    vectors=vectorize(train_d,train_vocab)
    '''for each word in vocabulary find the probability
    of word given target class for such all target classes'''
    words_prob=prob_words_given_class(train_vocab,vectors)
    #preprocessing of test data
    test_d,test_vocab=preprocess_doc(test)
    #list to hold the predicted target class value for test data
    prob_0=len([e for e in vectors if e[-1]==0])
    prob_1=len([e for e in vectors if e[-1]==1])
    test_target=[]
    for d in test_d:
        prob_0_given_test=prob_0
        prob_1_given_test=prob_1
        for word in train_vocab:
            prob_0_given_test*=words_prob[word][0]
            prob_1_given_test*=words_prob[word][1]
        if prob_0_given_test>prob_1_given_test:test_target.append([" ".join(d),0,[prob_0_given_test,prob_1_given_test]])
        elif prob_1_given_test>prob_0_given_test:test_target.append([" ".join(d),1,[prob_0_given_test,prob_1_given_test]])
    #printing of vocabulary
    print("No of Words in Vocabulary:",len(train_vocab))
    print("Vocabulary:")
    print(train_vocab)
    #printing of document and target classes
    for e in test_target:
        print("Test Document:",e[0])
        print("P(CV/Document):",e[2][0])
        print("P(DL/Document):",e[2][1])
        print("Target class:",class_dict[e[1]])


nb_multinomial()
