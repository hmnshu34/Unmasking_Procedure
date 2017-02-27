# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 16:22:29 2016

@author: root
"""

import os
import codecs
from sklearn import cross_validation
from sklearn import svm
from sklearn.utils import shuffle
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def getData(path, a,b):
    words = {}
    for filename in os.listdir(path + '/allc1'):
        with codecs.open(path + '/allc1/' + filename, "r",encoding='utf-8', errors='ignore') as file:
            if filename[0] == 'B' and a==1:
                words[file.read()] = 'Brooklyn1'
            elif filename[1] == 'e' and a==2:
                words[file.read()] = 'Herald1'
            elif filename[1] == 'Y' and a==3:
                words[file.read()] = 'Tribune1'
    for filename in os.listdir(path + '/allc2'):
        with codecs.open(path + '/allc2/' + filename, "r",encoding='utf-8', errors='ignore') as file:
            if filename[0] == 'B' and (a==4 or b ==4):
                words[file.read()] = 'Brooklyn2'
            elif filename[1] == 'e' and (a==5 or b ==5):
                words[file.read()] = 'Herald2'
            elif filename[1] == 'Y' and (a==6 or b ==6):
                words[file.read()] = 'Tribune2'
            elif b==0:
                words[file.read()] = 'Test'
        file.close()
    return list(words.keys()),list(words.values())
    

def getWords(data, numfeatures):

    count_vect = CountVectorizer()
    all_words_count_vectors = count_vect.fit_transform(data)
    word_freqs = [(word,all_words_count_vectors.getcol(idx).sum()) for word,idx in count_vect.vocabulary_.items()]
    sorted_words = sorted(word_freqs, key=lambda x:-x[1])
    
    return [x[0] for x in sorted_words[0:numfeatures]]
    


def CVScores(data, target, wordlist, clf):
    total = 0
    with np.errstate(invalid='ignore'):   
        for x in range(10):
            data, target = shuffle(data,target)
             
           #create a matrix for the data train
            count_vect = CountVectorizer(vocabulary=wordlist)
            X_train_counts = count_vect.fit_transform(data)

            total += cross_validation.cross_val_score(clf, X_train_counts, target, cv=10).mean()
#            predicted = cross_validation.cross_val_predict(clf,X_train_counts,target, cv=10)
    return total/10


def show_most_informative_features(vectorizer, clf, n):
    lista = []
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
#        print ("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))
        lista.append(fn_1), lista.append(fn_2)
    return lista
        
        
def most_informative_feature_for_binary_classification(vectorizer, classifier, n=10):
    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names()
    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
    topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]

    for coef, feat in topn_class1:
        print (class_labels[0], coef, feat)

    print()

    for coef, feat in reversed(topn_class2):
        print (class_labels[1], coef, feat)

def grafComp():
    methods = [MultinomialNB(),svm.SVC(kernel='linear', C=1),KNeighborsClassifier(n_neighbors=3)]
    c = ['red', 'blue', 'green', 'yellow', 'violet', 'orange']

    for m in methods:
        for a in range(1,4):
            lista, i = [], 0
            print('---------------------')
            print(str(m)[:20])
            for b in range(4,7):
                x,y = [],[]
                for numfeatures in range(50,1001,50):
                
                    data,target = getData(os.getcwd(), a,b)
                    
                    wordlist = getWords(data,numfeatures)
                        
                    scores= CVScores(data,target, wordlist, m)
                    
                    x.append(numfeatures), y.append(scores.mean())
                    
                plt.plot(x, y, color = c[i])
                i+=1
                lista.append(list(set(target))[0][:3] + list(set(target))[0][-1] + '/' + list(set(target))[1][:3] + list(set(target))[1][-1] )
            plt.legend(lista, loc='center left', bbox_to_anchor=(1, 0.85))
            plt.show()

def grafCompWord():         
    m = MultinomialNB()
    c = ['red', 'blue', 'green', 'yellow', 'violet', 'orange']
    
    for a in range(1,4):
        lista, i = [], 0
        print('---------------------')
        print(str(m)[:20])
        for b in range(4,7):
            x,y = [],[]
            data,target = getData(os.getcwd(), a,b)
            wordlist = getWords(data,350)
            for wordsRem in range(80):                    
                scores= CVScores(data,target, wordlist, m)
                
                x.append(wordsRem), y.append(scores.mean())
                
                count_vect = CountVectorizer(vocabulary=wordlist)
                X_train_counts = count_vect.fit_transform(data)
                clf = m.fit(X_train_counts,target)    
                
                mostF = show_most_informative_features(count_vect,clf, 1)
                wordlist.remove(mostF[0])
                wordlist.remove(mostF[1])
            plt.plot(x, y, color = c[i])
            i+=1
            lista.append(list(set(target))[0][:3] + list(set(target))[0][-1] + '/' + list(set(target))[1][:3] + list(set(target))[1][-1] )
        plt.legend(lista, loc='center left', bbox_to_anchor=(1, 0.85))
        plt.show()
        
def grafCompWordTes():    
    methods = [MultinomialNB()]
    c = ['red', 'blue', 'green', 'yellow', 'violet', 'orange']
    for m in methods:
        lista, i = [], 0
        print('---------------------')
        print(str(m)[:20])
        for a in range(1,4):
            b=0
            x,y = [],[]
            data,target = getData(os.getcwd(), a,b)
            wordlist = getWords(data,300)
            for wordsRem in range(80):                    
                scores= CVScores(data,target, wordlist, m)
                
                x.append(wordsRem), y.append(scores.mean())
                
                count_vect = CountVectorizer(vocabulary=wordlist)
                X_train_counts = count_vect.fit_transform(data)
                clf = m.fit(X_train_counts,target)    
                
                mostF = show_most_informative_features(count_vect,clf, 1)
                wordlist.remove(mostF[0])
                wordlist.remove(mostF[1])
            plt.plot(x, y, color = c[i])
            i+=1
            lista.append(list(set(target))[0][:3] + list(set(target))[0][-1] + '/' + list(set(target))[1][:3] + list(set(target))[1][-1] )
        plt.legend(lista, loc='center left', bbox_to_anchor=(1, 0.85))
        plt.show()
    
def grafCompTes():
    methods = [MultinomialNB(),svm.SVC(kernel='linear', C=1),KNeighborsClassifier(n_neighbors=3)]
    c = ['red', 'blue', 'green', 'yellow', 'violet', 'orange']
    
    for m in methods:
        lista, i = [], 0
        print('---------------------')
        print(str(m)[:20])
        for a in range(1,4):
            b = 0
            x,y = [],[]
            for numfeatures in range(50,1001,50):
            
                data,target = getData(os.getcwd(), a,b)
                
                wordlist = getWords(data,numfeatures)
                    
                scores= CVScores(data,target, wordlist, m)
                
                x.append(numfeatures), y.append(scores)
                    
            plt.plot(x, y, color = c[i])
            i+=1
            lista.append(list(set(target))[0][:3] + list(set(target))[0][-1] + '/' + list(set(target))[1][:3] + list(set(target))[1][-1] )
        plt.legend(lista, loc='center left', bbox_to_anchor=(1, 0.85))
        plt.show()
        
grafComp()