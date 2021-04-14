#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib.pyplot as pp
def graph(parameters, errors):
  pp.figure()
  pp.plot(parameters, errors, 'o') # g for green, o for circle
  pp.xlabel('hyperparameters')
  pp.ylabel('errors')
  pp.show() # This command will open the figure, and wait

# In[ ]:

