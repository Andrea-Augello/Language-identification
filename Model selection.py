#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Project: language identification

# ### Load libraries

from __future__ import print_function
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import string
import collections
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


# ## Model selection

# ### Load dataset
# We generate three separate DataFrames due to memory constraints, else we incurr into a `MemoryError`, the starting data is in the format:


#get_ipython().system('tail ./Data/*.txt')


print("Loading data...")

names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
         'U', 'V', 'W','X', 'Y', 'Z', 'class']

dataset_it = pd.DataFrame((parse_string(word,'Italian') for word in open('./Data/italian.txt', encoding = "ISO-8859-1")), columns = names)
dataset_en = pd.DataFrame(([parse_string(word,'English') for word in open('./Data/english.txt', encoding = "ISO-8859-1")]), columns=names)
dataset_zh = pd.DataFrame(([parse_string(word,'Chinese') for word in open('./Data/chinese.txt', encoding = "ISO-8859-1")]) , columns = names)


# Here we join the data from the three datasets

dataset = pd.DataFrame()
dataset = dataset.append(dataset_it)
dataset = dataset.append(dataset_en)
dataset = dataset.append(dataset_zh)


#  ### Visualization on the different letter frequency in the analyzed languages

 # Code used to generate the histograms
dataset_it.hist()
dataset_en.hist()
dataset_zh.hist()

pyplot.show()

array = dataset.values
X = array[:,0:26]
Y = array[:,26]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,test_size=validation_size, random_state=seed, shuffle=True)


# ### Spot-Check Algorithms
# Here we train many different models with the training set as to compare their performances and pick the most promising one to use


models=[]
models.append(('LR', LogisticRegression(multi_class='auto', solver='liblinear')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='scale')))
models.append(('MLP', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30,10), random_state=3)))




# Evalutate each model in turn

print("Evalutating different models...")

results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed) 
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# ### Compare Algorithms

# Generating the boxplots 
fig = pyplot.figure()
fig.suptitle('Algorithm comparison')
ax = fig.add_subplot(111)
ax.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# #### Select a model (Support Vector) and train it with all the training data
# We then make prediction for the validation set to extimate the final model's accuracy



print("Training final model...")

model = SVC(gamma='scale', probability=True)
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)


# ### Final accuracy analysis 


model_name = str(model).split('(')[0]
print("Model:",model_name,"\n")
print("Accuracy score:\n",accuracy_score(Y_validation, predictions))
print("\n\nConfusion matrix:\n")
data = {'y_Predicted': predictions,
        'y_Actual':    Y_validation
        }

df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])

confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)
print("\n\nClassification report:\n",classification_report(Y_validation, predictions))


# ### Save current model to file

#filename = 'finalized_model.sav'
#pickle.dump(model, open(filename, 'wb'))
#print("Saved", model)
