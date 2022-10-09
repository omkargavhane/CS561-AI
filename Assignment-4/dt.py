import csv
import re
import nltk
from pprint import pprint
import numpy as np
import math
from nltk.corpus import stopwords
import string
from termcolor import colored

#tempaltes for printing 
print_rev_green = lambda x : print(colored(x,'green',attrs=['reverse','bold']))
print_rev_red = lambda x : print(colored(x,'red',attrs=['reverse','bold']))

def print_dataset(data,row):
    for i in range(row):
        print_rev_red('Row '+str(i))
        for attr in data[i]:
            print(attr)

def create_dataset(fname):
    #reading dataset
    lol=list(csv.reader(open(fname), delimiter=':'))
    target_class=list(set([e[0] for e in lol]))
    target_class.sort()
    target_class_map=dict(zip(target_class,range(len(target_class))))
    print_rev_green("Mapping for target class")
    print(target_class_map) 
    #Mapping target class to numeric value
    data=[[target_class_map[row[0]]," ".join(row[1].split()[1:]),None,None,None,None] for row in lol]
    #removing punctuation and numeric value from question and generating vocabulary
    vocabulary=[]
    for row in data:
        questn=[word.lower() for word in re.sub('[^a-zA-Z]', ' ',row[1]).split() if len(word)>1]
        row[2]=questn
        row[3]=len(questn)
        vocabulary.extend(questn)
    #creating ngram 
    ngram=list(nltk.ngrams(vocabulary,1))
    #counting a ngram
    ngram_frequency={}
    for e in ngram:
        if " ".join(e) in ngram_frequency:ngram_frequency[" ".join(e)]+=1
        else:ngram_frequency[" ".join(e)]=1
    #sorting in decreasing order 
    ngram_frequency=sorted(ngram_frequency.items(),key=lambda x:x[1],reverse=True)
    #selecting top 500 from sorted list
    most_frequent_500=[e[0] for e in ngram_frequency[:500]]
    #converting ngram_frequency again back to dictionary from list of tuples
    ngram_frequency=dict(ngram_frequency)
    print_rev_green("Datatset Attribute")
    print("-"*120)
    print("| Targte class | Raw Question | Question After preprocessing | length of question | Lexical | syntactical |")
    print("-"*120)
    print_rev_green("first 5 rows from dataset after processing the question")
    print_dataset(data,5)
    #extracting lexical and syntactic data
    for row in data:
        lexical=[]
        syntax=[]
        for word in row[2]:
            if word in most_frequent_500:
                lexical.append(word)
                syntax.append(nltk.pos_tag([word])[0][1])
        row[4]=lexical
        row[5]=syntax
    print_rev_green("first 5 rows from dataset after extracting lexical and syntactic data")
    print_dataset(data,5)
    return target_class_map,ngram_frequency,most_frequent_500,data

def prob_word_given_target_class(data,word,column_no,target_class,vocab_length):
    target_class_rows=[questn for questn in data if questn[0]==target_class]
    count_word_given_target_class=sum([row[column_no].count(word) for row in target_class_rows])
    return (count_word_given_target_class+1)/(len(target_class_rows)+vocab_length)

def prob_words_given_target_classes(target_classes,column_no,data,words):
    map_prob_word_given_target_class={}
    for word in words:
        map={}
        for target_class in target_classes:
            map[target_class]=prob_word_given_target_class(data,word,column_no,target_class,len(words))
        map_prob_word_given_target_class[word]=map
    return map_prob_word_given_target_class

def train_calculate_column_probabilities(data,mpwgtc,mppgtc):
    for row in data:
        row[4]=np.product([mpwgtc[word][row[0]] for word in row[4]])
        row[5]=np.product([mppgtc[word][row[0]] for word in row[5]])
    print_rev_green("first 5 rows from training dataset after probability calculations")
    print_dataset(data,5)
    return data


def test_calculate_column_probabilities(data,mpwgtc,mppgtc):
    for row in data:
        row[4]=np.product([max(mpwgtc[word].values()) if word in mpwgtc else 1 for word in row[4]])
        row[5]=np.product([max(mppgtc[word].values()) if word in mppgtc else 1 for word in row[5]])
    print_rev_green("first 5 rows from testing dataset after probability calculations")
    print_dataset(data,5)
    return data

def best_split(data,algo):
    transpose_data=np.array(data).T.tolist()
    impurity_attribute=[]
    #print(len(data))
    for i in range(1,len(transpose_data)):
        data=transpose_data[i]
        data_target=list(zip(data,transpose_data[0]))
        unique_data=list(set(data))
        unique_data.sort()
        #print("unique_data",unique_data)
        split_position=[(unique_data[i]+unique_data[i+1])/2 for i in range(len(unique_data)-1)]
        if len(unique_data)==1:
            split_position=[unique_data[0]]
        #print("split positions",split_position)
        impurity=[]
        min_gini_split=1
        for position in split_position:
            le=[]
            le_target_frequency={}
            gt=[]
            gt_target_frequency={}
            for row in data_target:
                if row[0]<=position:le.append(row)
                else:gt.append(row)
            for target in target_class_map.values():
                le_target_frequency[target]=0
                gt_target_frequency[target]=0
            for row in le:
                le_target_frequency[row[1]]+=1
            for row in gt:
                gt_target_frequency[row[1]]+=1
            le.append(None)
            gt.append(None)
            #gini
            if algo=="gini":
                algo_le=1-sum([(e[1]/len(le))**2 for e in le_target_frequency.items()])
                algo_gt=1-sum([(e[1]/len(gt))**2 for e in gt_target_frequency.items()])
                algo_split=(len(le)/(len(le)+len(gt)))*algo_le+(len(gt)/(len(le)+len(gt)))*algo_gt
            #entropy
            if algo=="entropy":
                algo_le=sum([-1*(e[1]/len(le))*math.log2((e[1]+1)/len(le)) for e in le_target_frequency.items()])
                algo_gt=sum([-1*(e[1]/len(gt))*math.log2((e[1]+1)/len(gt)) for e in gt_target_frequency.items()])
                algo_split=(len(le)/(len(le)+len(gt)))*algo_le+(len(gt)/(len(le)+len(gt)))*algo_gt
            #classification error
            if algo=="ce":
                algo_le=1-max([e[1]/len(le) for e in le_target_frequency.items()])
                algo_gt=1-max([e[1]/len(gt) for e in gt_target_frequency.items()])
                algo_split=(len(le)/(len(le)+len(gt)))*algo_le+(len(gt)/(len(le)+len(gt)))*algo_gt
            target_class=None
            '''
            if int(algo_split)==0:
                for e in target_class_map.values():
                    if le_target_frequency[e]+gt_target_frequency[e]==len(data_target):
                        target_class=e
            '''
            target_frequency=[[e,le_target_frequency[e]+gt_target_frequency[e]] for e in target_class_map.values()]
            target_class=sorted(target_frequency,key=lambda x:x[1],reverse=True)[0][0]
            impurity.append([i,position,algo_split,target_class])
        #print("impurity",impurity)
        #print("target_class",target_class)
        impurity_attribute.append(sorted(impurity,key=lambda x:x[2])[0])
        #print("impurity attribute",impurity_attribute)
        #if target_class:break
    return sorted(impurity_attribute,key=lambda x:x[2])[0]

class node:
    def __init__(self,data):
        self.attr=None
        self.val=None
        self.impurity=None
        self.data=data
        self.left=None
        self.right=None
        self.target_class=None

def build_tree(train_data,algo):
    start=node(train_data)
    level=[start]
    cnt=0
    while level:
        cur=level[0]
        #if len(cur.data)>1:
        decision=best_split(cur.data,algo)
        cur.attr=decision[0]
        cur.val=decision[1]
        cur.impurity=decision[2]
        cur.target_class=decision[3]
        #if len(cur.data)==1:
        #cur.target_class=cur.data[0][0]
        #cur.impurity=0
        if cur.impurity!=0:
            left=[row for row in cur.data if row[cur.attr]<=cur.val]
            right=[row for row in cur.data if row[cur.attr]>cur.val]
            if left:
                cur.left=node(left)
                level.append(cur.left)
            if right:
                cur.right=node(right)
                level.append(cur.right)
        level.pop(0)
        #cnt+=1
        #if cnt==20000:
        #    break
    return start

def dt_classifier(start,attributes):
    cur=start
    target_class=None
    while cur and cur.attr:
        #print("current attribute",cur.attr)
        if attributes[cur.attr-1]<=cur.val:
            target_class=cur.target_class
            cur=cur.left
        else:
            target_class=cur.target_class
            cur=cur.right
    return target_class

#creation of train dataset
target_class_map,train_ngram_frequency,train_most_frequent_500,train_data=create_dataset("dt_train.csv")
#uniques pos from train data
unique_pos=set([e for row in train_data for e in row[-1]])
#mapper for word given all target classes(lexical)
map_prob_word_given_target_class=prob_words_given_target_classes(sorted(target_class_map.values()),4,train_data,train_most_frequent_500)
#mapper for pos given all target classes(syntax)
map_prob_pos_given_target_class=prob_words_given_target_classes(sorted(target_class_map.values()),5,train_data,unique_pos)
#calculate the probabilities for lexical and syntactical attribute
train_data_after_prob_cal=train_calculate_column_probabilities(train_data,map_prob_word_given_target_class,map_prob_pos_given_target_class)
#selecting only target class,length,lexical,synatax attribute
train_data=[[row[0],row[3],row[4],row[5]] for row in train_data_after_prob_cal]

#creation of test dataset
target_class_map,test_ngram_frequency,test_most_frequent_500,test_data=create_dataset("dt_test.csv")
#unique pos from test data
unique_pos=set([e for row in test_data for e in row[-1]])
#calculate the probabilities for lexical and syntactical attribute
test_data_after_prob_cal=test_calculate_column_probabilities(test_data,map_prob_word_given_target_class,map_prob_pos_given_target_class)

actual_target=np.array(test_data_after_prob_cal).T.tolist()[0]

test_predicted_algo=[]
for algo in ["gini","entropy","ce"]:
    print_rev_red(algo)
    #build tree
    print_rev_green("Building Tree ...")
    start=build_tree(train_data,algo)
    #prediction
    print_rev_green("Predicting test data ...")
    test_predicted=[]
    for questn in test_data_after_prob_cal:
        test_predicted.append(dt_classifier(start,[questn[3],questn[4],questn[5]]))
    cnt=0
    for i in range(len(actual_target)):
        if actual_target[i]==test_predicted[i]:
            cnt+=1
    print("Accuracy",cnt/len(test_predicted)*100)
    #calculating precision,recall,fscore
    from sklearn.metrics import precision_recall_fscore_support
    metric=precision_recall_fscore_support(actual_target,test_predicted,labels=range(len(target_class_map)))
    print_rev_green(" "*10+"| ".join([e[0]+" "+str(e[1]) for e in target_class_map.items()]))
    metric_name=['precision','recall','f-score','support']
    for i  in range(len(metric)):
        print(metric_name[i],metric[i].tolist())
    test_predicted_algo.append(test_predicted)
#Observe how many samples are mis-classified using gini index based
#model but correctly classified by mis-classification error and
#cross-entropy based model.
false_gini_true_entropy=0
false_gini_true_ce=0
for i in range(len(test_predicted)):
    if test_predicted_algo[0][i]!=actual_target[i] and test_predicted_algo[1][i]==actual_target[i]:
        false_gini_true_entropy+=1
    if test_predicted_algo[0][i]!=actual_target[i] and test_predicted_algo[2][i]==actual_target[i]:
        false_gini_true_ce+=1
print("samples misclassfied by gini but correctly classified by entropy",false_gini_true_entropy)
print("samples misclassfied by gini but correctly classified by ce",false_gini_true_ce)
