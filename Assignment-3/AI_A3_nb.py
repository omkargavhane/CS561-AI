import csv
import re
import nltk
import math
import numpy as np
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
k=5

#read data
lol=list(csv.reader(open('sms_spam.csv'), delimiter='\t'))
targets=[l[0] for l in lol]
print(targets)
msgs=[l[1] for l in lol]
#lemmatization
lemmatizer = WordNetLemmatizer()
corpus=[]
for i in range(len(msgs)):
        review=re.sub('[^a-zA-Z]', ' ',msgs[i])
        review=review.lower()
        review=review.split()
        review=[lemmatizer.lemmatize(word) for word in review if word not in stopwords.words('english')]
        review=' '.join(review)
        corpus.append(review)

#splitting dataset creating folds
x_folds=[]
y_folds=[]
fl=fold_length=math.floor(len(corpus)/k)
start=0
for i in range(k-1):
    x_folds.append(list(corpus[start:fold_length]))
    y_folds.append(list(targets[start:fold_length]))
    start=fold_length
    fold_length+=fl
x_folds.append(list(corpus[start:]))
y_folds.append(list(targets[start:]))


#Multivariate model
vocab=vectorizer.fit_transform(corpus).toarray().tolist()

#Mapping target to 1/0 and adding target column to dataset
#1->"ham",0->"spam"
map_targets=[1 if e=="ham" else 0 for e in targets]
for i in range(len(vectors)):
    vectors[i].append(map_targets[i])

#testing 
for i in range(k):
    #current test is ith fold
    x_test=x_folds[i]
    y_test=y_folds[i]
    #except ith all are training data
    x_train=[]
    y_train=[]
    for j in range(k):
        if j!=i:
            x_train.extend(x_folds[j])
            y_train.extend(y_folds[j])
    #list to hold predicted value 
    test_target=[]
    #vectors with target class 1
    target_1=[sentence for sentence in train if sentence[-1]==1]
    #calculate tarnspose of it 
    target_1_transpose=np.array(target_1).transpose().tolist()
    #no of vectors with target class 1
    target_1_length=len(target_1)
    #probability of target class 1
    target_prob_1=target_1_length/len(train)
    #vectors with target class 0
    target_0=[sentence for sentence in train if sentence[-1]==0]
    #calculate tarnspose of it 
    target_0_transpose=np.array(target_0).transpose().tolist()
    #no of vectors with target class 0
    target_0_length=len(target_0)
    #probability of target class 0
    target_prob_0=target_0_length/len(train)
    #iterate over sentences in train data
    for sentence in test:
        prob_word_given_1=[]
        prob_word_given_0=[]
        for w in range(len(sentence)-1):
            if sentence[w]:
                count_1=target_1_transpose[w].count(1)
                if count_1==0:
                    prob_word_given_1.append(1/target_1_length+len(vectors[0]))
                    continue
                prob_word_given_1.append(count_1/target_1_length)
            else:
                count_0=target_0_transpose[w].count(0)
                if count_0==0:
                    prob_word_given_0.append(1/target_0_length+len(vectors[0]))
                    continue
                prob_word_given_0.append(count_0/target_0_length)
        prob_1_given_sentence=target_prob_1*math.prod(prob_word_given_1)
        prob_0_given_sentence=target_prob_0*math.prod(prob_word_given_0)
        if prob_1_given_sentence > prob_0_given_sentence:
            test_target.append(1)
        else:
            test_target.append(0)
    num=0
    for s in range(len(test)):
        if test[s][-1]==test_target[s]:num+=1
    print("Fold",i,"Accuracy",num/len(test))

Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray().tolist()'''
