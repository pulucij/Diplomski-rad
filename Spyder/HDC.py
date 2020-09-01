# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 17:13:47 2020

@author: Lucija
"""

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

import random

from sklearn.metrics import balanced_accuracy_score

def KNN_Classifier(X_train_features, X_test_features, y_train, y_test, n):
    classifier = KNeighborsClassifier(n_neighbors = n, metric = 'minkowski', p = 2)
    classifier.fit(X_train_features, y_train)
    
    y_pred = classifier.predict(X_test_features)
    
    # Making the Confusion Matrix
    #cm = confusion_matrix(y_test, y_pred)
    
    # Making the Confusion Matrix
    #cm = confusion_matrix(y_test, y_pred)
    
    return y_pred

def RF_Classifier(X_train_features, X_test_features, y_train, y_test, n):
    classifier = RandomForestClassifier(n_estimators = n, random_state = 0)
    classifier.fit(X_train_features, y_train)
    
    y_pred = classifier.predict(X_test_features)
    
    # Making the Confusion Matrix
    #cm = confusion_matrix(y_test, y_pred)
    
    # Making the Confusion Matrix
    #cm = confusion_matrix(y_test, y_pred)
    
    return y_pred

def NB_Classifier(X_train_features, X_test_features, y_train, y_test):
    classifier = GaussianNB()
    classifier.fit(X_train_features, y_train)
    
    y_pred = classifier.predict(X_test_features)
    
    return y_pred

def calculate_accuracy(bee, X_train, X_test, y_train, y_test, algorithm, n):
    #count number of features
    k=0
    for j in range(0,len(bee)):
        if bee[j]==1:
            k+=1
    X_train_features = np.zeros((X_train.shape[0], k))
    X_test_features = np.zeros((X_test.shape[0], k))
        
    k=0
    for j in range(0,len(bee)):
        if bee[j]==1:
            X_train_features[:,k]=X_train[:,j]
            X_test_features[:,k]=X_test[:,j]
            k+=1
    #print(X_train_features)
    #print(X_test_features)
                
    # Fitting K-NN to the Training set
    #minkowski sa p=2 je euklidska
    #classifier = KNeighborsClassifier(n_neighbors = 15, metric = 'minkowski', p = 2)
    #classifier.fit(X_train_features, y_train)
    
    # Training the Random Forest Classification model on the Training set
    #from sklearn.ensemble import RandomForestClassifier
    #classifier = RandomForestClassifier(n_estimators = 5, criterion = 'entropy', random_state = 0)
    #classifier.fit(X_train_features, y_train)
    #0.9043478260869565
    #[0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    
    
    
    #y_pred = classifier.predict(X_test_features)
    
    # Making the Confusion Matrix
    """if(algorithm == 'KNN'):
        cm = KNN_Classifier(X_train_features, X_test_features, y_train, y_test, n)
    elif(algorithm == 'RF'):
        cm = RF_Classifier(X_train_features, X_test_features, y_train, y_test, n)
    else:
        raise ValueError('Key word is not recognized.')"""

    
    #print(cm)
        
    #cm_columns = cm.shape[1] 
    #tp_tn=0
    #fp_fn=0
    #Calculating TP+TN i FP+FN
    #for j in range(0,cm_columns):
     #   fp_fn += cm.sum(axis=0)[j]+cm.sum(axis=1)[j]-2*cm[j][j]
      #  for k in range(0, cm_columns):
       #     tp_tn+=np.sum(cm[k])
            
    #print(tp_tn)    
    #print(fp_fn)
    #tp_tn-=fp_fn 
        
                
    #return tp_tn/(tp_tn+fp_fn)
    if(algorithm == 'KNN'):
        y_pred = KNN_Classifier(X_train_features, X_test_features, y_train, y_test, n)
    elif(algorithm == 'RF'):
        y_pred = RF_Classifier(X_train_features, X_test_features, y_train, y_test, n)
    elif(algorithm == 'NB'):
        y_pred = NB_Classifier(X_train_features, X_test_features, y_train, y_test)
    else:
        raise ValueError('Key word is not recognized.')
        
    return balanced_accuracy_score(y_test, y_pred)



def init(number_employed_bees, columns_number, employed_matrix):
    for i in range(0,number_employed_bees):
        test=0
        while test==0:
            array = np.zeros(columns_number)
            for j in range(0,columns_number):
                rand_number = random.random()
                if rand_number <= 0.5:
                    array[j]=0
                else:
                    array[j]=1
            if not np.isin(1, array):
               continue
            for j in range(0,i):
                if np.array_equal(employed_matrix[j],array):
                    continue
            employed_matrix[i] = array
            test=1
            
def init_accuracies_and_best(number_employed_bees, accuracies, X_train, X_test, y_train, y_test, algorithm, n, best, best_bee, 
                             employed_matrix, columns_number):
    for i in range(0,number_employed_bees):
        accuracies[i]=calculate_accuracy(employed_matrix[i], X_train, X_test, y_train, y_test, algorithm, n)

    best = np.amax(accuracies)
    for i in range(number_employed_bees):
        if accuracies[i] == best:
            best_bee = [employed_matrix[i][k] for k in range(columns_number)]
            
def iterations(max_iterations_number, limit_array, limit, employed_matrix, number_employed_bees, columns_number, algorithm, n,
               accuracies, number_onlooker_bees, X_train, X_test, y_train, y_test, best, best_bee):
    for i in range(0,max_iterations_number):
        limit_array= [x+1 for x in limit_array]
        for j in range(0,number_employed_bees):
            bee = [employed_matrix[j][k] for k in range(columns_number)] #Ovde je bitno napravit novo
                                                                        #polje/listu, jer onako kad mjenjaš
                                                                        #bee, mjenjaš i employed_matrix!!!
            first = np.random.randint(0,columns_number)
            second = np.random.randint(0,columns_number) #Tu još razmotri ono sa sigmoid funkcijom!!!
            while first==second or bee[first]==bee[second]:
                second = np.random.randint(0,columns_number)
            #print(j) print(bee) print(accuracies[j])
            temp=bee[first]
            bee[first]=bee[second]
            bee[second]=temp
            bee_accuracy = calculate_accuracy(bee, X_train, X_test, y_train, y_test, algorithm, n)
            #print(bee) print(bee_accuracy)
            if bee_accuracy > accuracies[j]:
                    employed_matrix[j]=bee
                    accuracies[j]=bee_accuracy
                    limit_array[j]=0

        P = np.zeros(number_employed_bees)
        numerator = np.sum(accuracies)
        for j in range(0,number_employed_bees):
            P[j]=accuracies[j]/numerator

        #Tu nešto ne valja u ovom polju!!! Trebat ce mozda neka lista
        #To si valjda popravila...
        onlooker_bees_indexes = np.zeros(number_onlooker_bees)
        k=0    
        for j in range(0,number_onlooker_bees):
            pick = random.random() #ili random.uniform(0,1)??
            if pick < P[0]:
                onlooker_bees_indexes[k]=0
                k+=1
                #print(pick) print(k)
                continue
            for l in range(1,number_employed_bees):
                if pick < np.sum(P[0:l+1]) and pick >= np.sum(P[0:l]):
                    onlooker_bees_indexes[k]=l
                    k+=1
                    break
                #print(np.sum(P[0:l+1])) print(np.sum(P[0:l])) print(pick) print(k)

        for j in onlooker_bees_indexes:
            bee = [employed_matrix[int(j)][k] for k in range(columns_number)]
            first = np.random.randint(0,columns_number)
            second = np.random.randint(0,columns_number)
            while first==second or bee[first]==bee[second]:
                second = np.random.randint(0,columns_number)
            #print(j) print(bee) print(accuracies[int(j)])
            temp=bee[first]
            bee[first]=bee[second]
            bee[second]=temp
            bee_accuracy = calculate_accuracy(bee, X_train, X_test, y_train, y_test, algorithm, n)
            #print(bee) print(bee_accuracy)
            if bee_accuracy > accuracies[int(j)]:
                    employed_matrix[int(j)]=bee
                    accuracies[int(j)]=bee_accuracy
                    limit_array[int(j)]=0

        for j in range(0,number_employed_bees):
            if limit_array[j] >= limit:
                test=0      
                while test==0:
                    array = np.zeros(columns_number)
                    for k in range(0,columns_number):
                        rand_number = random.random()
                        if rand_number <= 0.5:
                            array[k]=0
                        else:
                            array[k]=1
                    if not np.isin(1, array):
                        continue
                    for k in range(0,number_employed_bees):
                        if k!=j and np.array_equal(employed_matrix[k],array):
                            continue
                    employed_matrix[j] = array
                    limit_array[j]=0
                    accuracies[j]=calculate_accuracy(array, X_train, X_test, y_train, y_test, algorithm, n)
                    test=1

        #print(best) print(best_bee)           
        best_ = np.amax(accuracies)
        if best_ > best:
            best = best_
            for j in range(number_employed_bees):
                if accuracies[j] == best:
                    best_bee = [employed_matrix[j][k] for k in range(columns_number)] #Tu isto treba tako
                                                                                      #inače se best_bee mjenja kad i employed_matrix[i]  
                    break
        #print(best) print(best_bee) 
        print('---------'+str(i)+'---------')
        print(best)
        print(best_bee)
    
    return best, best_bee

#Import dataset
dataset = pd.read_csv('processed.cleveland.data', header=None, delimiter=',')
dataset[11] = pd.to_numeric(dataset[11], errors='coerce') 
dataset[12] = pd.to_numeric(dataset[12], errors='coerce') 
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

# Taking care of missing data
# Updated Imputer
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
missingvalues = missingvalues.fit(X)
X=missingvalues.transform(X)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)


#BABC
columns_number = X.shape[1]
number_employed_bees = 12
number_onlooker_bees = 12
max_iterations_number = 200
limit = 10
BEST = 0
BEST_BEE = np.zeros(columns_number)


employed_matrix = np.zeros((number_employed_bees,columns_number))
limit_array = np.zeros(number_employed_bees)
accuracies = np.zeros(number_employed_bees)
best = 0
best_bee = np.zeros(columns_number)

#KNN    
init(number_employed_bees, columns_number, employed_matrix)
    
init_accuracies_and_best(number_employed_bees, accuracies, X_train, X_test, y_train, y_test, 'KNN', 5, best, best_bee, 
                             employed_matrix, columns_number)
    
RESULT = iterations(max_iterations_number, limit_array, limit, employed_matrix, number_employed_bees, columns_number, 'KNN', 5, 
                accuracies, number_onlooker_bees, X_train, X_test, y_train, y_test, best, best_bee)

#NB
#init_accuracies_and_best(number_employed_bees, accuracies, X_train, X_test, y_train, y_test, 'NB', 0, best, best_bee, 
#                             employed_matrix, columns_number)
    
#RESULT = iterations(max_iterations_number, limit_array, limit, employed_matrix, number_employed_bees, columns_number, 'NB', 0, 
#                accuracies, number_onlooker_bees, X_train, X_test, y_train, y_test, best, best_bee)

#RF
#init_accuracies_and_best(number_employed_bees, accuracies, X_train, X_test, y_train, y_test, 'RF', 10, best, best_bee, 
#                             employed_matrix, columns_number)
    
#RESULT = iterations(max_iterations_number, limit_array, limit, employed_matrix, number_employed_bees, columns_number, 'RF', 10, 
#                accuracies, number_onlooker_bees, X_train, X_test, y_train, y_test, best, best_bee)


if (RESULT[0] > BEST):
    BEST = RESULT[0]
    BEST_BEE = RESULT[1]
        