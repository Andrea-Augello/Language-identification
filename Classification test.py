#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Project: language identification

# ### Load libraries

from __future__ import print_function
import sys
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import collections
import string
import pickle


# ## Parsing functions

# ### Removing stresses
# In order to work with only ASCII characters for the chinese language _*pinyin*_ is used instead of _*hanzi*_, and every in every word stresses are removed.  

def strip_stress(word):
    table = collections.defaultdict(lambda: None)
    table.update({
        ord('é'): 'e',
        ord('ô'): 'o',
        ord('è'): 'e',
        ord('à'): 'a',
        ord('ì'): 'i',
        ord('ù'): 'u',
        ord('\n'): '',
    })
    table.update(dict(zip(map(ord,string.ascii_uppercase), string.ascii_lowercase)))
    table.update(dict(zip(map(ord,string.ascii_lowercase), string.ascii_lowercase)))
    table.update(dict(zip(map(ord,string.digits), string.digits)))
    return word.translate(table,)


# ### Text to feature vector
# This function converts every word into a feature vector where each feature is the amout of times a certain letter appears, if `scale` is set to `True` then the features will be scaled by dividing them by the total length of the word. 


def parse_string(word, lang = None, scale = False):
   
    str(word)
    word = strip_stress(word.lower())
    length = len(word)
    LetterFreq={}
    for letter in string.ascii_lowercase:
        LetterFreq[letter] = 0
    for letter in word.lower():
        LetterFreq[letter] += 1
    features = list(LetterFreq.values())
    
    if(length < 1 or scale == False):
        features = [float(x) for x in features]
    else:
        features = [float(x)/length for x in features]
    
    if(lang != None):
        features.append(lang)
    
    return features



# Load saved model
filename = 'finalized_model.sav'
model = pickle.load(open(filename, 'rb'))
print("Loaded model:\n", model)

# Run tests
print("\nInsert word to test:")

for line in sys.stdin:
	test_string = line

	classes = ["Chinese","English", "Italian"]

	predictions = model.predict_proba([parse_string(test_string)]).tolist()[0]
	best_guess = predictions.index(max(predictions))

	print("The word", test_string,  "was classified as",  classes[best_guess] )
	for i in range (0,3):
		print("\t"+classes[i]+":",round(predictions[i]*100, 3),"%")
	print("\n\nInsert word to test:")

