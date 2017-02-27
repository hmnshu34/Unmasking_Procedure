# -*- coding: utf-8 -*-

import os
import codecs

from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.filterwarnings('ignore')


def getData(path, directory_training, directory_target):
    words = {}
        
    for foldername in os.listdir(path):
        
        if os.path.isdir(foldername) and (foldername in directory_training or foldername in directory_target) :
                
                for filename in os.listdir(path+"\\"+foldername):
                   
                    if (filename.endswith(".txt")):
                            
                            with codecs.open(path +"\\"+foldername+"\\"+ filename, "r",encoding='utf-8', errors='ignore') as file:
                            
                                words[file.read()] = foldername
          		
    return list(words.keys()),list(words.values())
    

def getWords(data, numfeatures):

    count_vect = CountVectorizer()
    all_words_count_vectors = count_vect.fit_transform(data) #Learn the vocabulary dictionary and return term-document matrix
	
	#create a dictionary with every word 
	#in the term-document matrix matched with its frequency
    word_freqs = [(word,all_words_count_vectors.getcol(idx).sum()) for word,idx in count_vect.vocabulary_.items()] 
    '''In this case, sorted_words will return the element in that array whose second element 
	(x[1]) is larger than all of the other elements' second elements.'''
	
    sorted_words = sorted(word_freqs, key=lambda x:x[1], reverse=True)
	
    return [x[0] for x in sorted_words[0:numfeatures]] #return the words with the highest frequencies
    
