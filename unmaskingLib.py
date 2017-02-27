# -*- coding: utf-8 -*-

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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfTransformer

import warnings
warnings.filterwarnings('ignore')


def CVScores(data, target, wordlist, classifier, averaging_cvscores_times):
    total = 0
	
    with np.errstate(invalid='ignore'):   
        for x in range(averaging_cvscores_times): 
            data, target = shuffle(data, target)
             
			#create a matrix for the data train
            count_vect = CountVectorizer(vocabulary=wordlist)
            X_train_counts = count_vect.fit_transform(data)
			
			#comparison between the target and the training set thanks to the cross validation
            total += cross_validation.cross_val_score(classifier, X_train_counts, target, cv=10).mean() 
           					
    return float(total)/float(averaging_cvscores_times)


def show_most_informative_features(vectorizer, classifier, n):

    list_of_features = []
    feature_names = vectorizer.get_feature_names() #Convert a collection of raw documents to a matrix of TF-IDF features
	
    coefs_with_features_names_sorted = sorted(zip(classifier.coef_[0], feature_names)) #Features numbers are sorted here 
    top = zip(coefs_with_features_names_sorted[:n], coefs_with_features_names_sorted[:-(n + 1):-1])
	
    for (coef_1, fn_1), (coef_2, fn_2) in top:
		
        list_of_features.append(fn_1), list_of_features.append(fn_2)
		
    return list_of_features
        