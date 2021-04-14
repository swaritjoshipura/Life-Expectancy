#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
def training_validation_testing(data, labels):
  positive_samples = np.where(labels == 1)[0]
  negative_samples = np.where(labels == -1)[0]
  zero_samples = np.where(labels == 0)[0]

  train_samples = list(positive_samples[0:int(np.floor(len(positive_samples)/3))]) + list(negative_samples[0:int(np.floor(len(negative_samples)/3))]) + list(zero_samples[0:int(np.floor(len(zero_samples)/3))])  

  validation_samples = list(positive_samples[int(np.floor(len(positive_samples)/3)):int(np.floor(2*len(positive_samples)/3))]) + list(negative_samples[int(np.floor(len(negative_samples)/3)):int(np.floor(2*len(negative_samples)/3))]) + list(zero_samples[int(np.floor(len(zero_samples)/3)):int(np.floor(2*len(zero_samples)/3))])
  

  test_samples = list(positive_samples[int(np.floor(2*len(positive_samples)/3)):len(positive_samples)]) + list(negative_samples[int(np.floor(2*len(negative_samples)/3)):len(negative_samples)]) + list(zero_samples[int(np.floor(2*len(zero_samples)/3)):len(zero_samples)])

  k_list = [0, 1, 2, 3]
  errors = []
  best_err = 1.1
  best_k = 0
  for k in k_list:
    gnb = 0
    if k == 0:
      gnb = GaussianNB()
    elif k == 1:
      gnb = MultinomialNB() 
    elif k == 2:
      gnb = ComplementNB()
    elif k == 3:
      gnb = BernoulliNB()
    

    
    labels_pred = gnb.fit(data[train_samples], labels[train_samples]).predict(data[validation_samples])
    err = np.mean(labels[validation_samples] != np.array([labels_pred]).T)
    errors.append(err)
    
    if k == 0:
      print("Algorithm=Gaussian Naive-Bayes" + " err=" + str(err))
    elif k == 1:
      print("Algorithm=Multinomial Naive-Bayes" + " err=" + str(err))
    elif k == 2:
      print("Algorithm=Complement Naive-Bayes" + " err=" + str(err))
    elif k == 3:
      print("Algorithm=Bernoulli Naive-Bayes" + " err=" + str(err))

    if err < best_err:
      best_err = err
      best_k = k


  if best_k == 0:
    print("best algorithm = Gaussian Naive-Bayes")
    gnb = GaussianNB()
    labels_pred = gnb.fit(data[train_samples], labels[train_samples]).predict(data[test_samples])
    err = np.mean(labels[test_samples] != np.array([labels_pred]).T)
    print("final best err = " + str(err))
  elif best_k == 1:
    print("best algorithm = Multinomial Naive-Bayes")
    gnb = MultinomialNB()
    labels_pred = gnb.fit(data[train_samples], labels[train_samples]).predict(data[test_samples])
    err = np.mean(labels[test_samples] != np.array([labels_pred]).T)
    print("final best err = " + str(err))
  elif best_k == 2: 
    print("best algorithm = Complement Naive-Bayes")
    gnb = ComplementNB()
    labels_pred = gnb.fit(data[train_samples], labels[train_samples]).predict(data[test_samples])
    err = np.mean(labels[test_samples] != np.array([labels_pred]).T)
    print("final best err = " + str(err))
  elif best_k == 3:
    print("best algorithm = Bernoulli Naive-Bayes") 
    gnb = BernoulliNB()
    labels_pred = gnb.fit(data[train_samples], labels[train_samples]).predict(data[test_samples])
    err = np.mean(labels[test_samples] != np.array([labels_pred]).T)
    print("final best err = " + str(err))
 
  
  return k_list, errors
  
# In[ ]:

