
# coding: utf-8

# In[2]:

import numpy as np

def vote(a,b):
    result = np.random.randint(0,2)
    if result == 0:
        return "오늘의 발표자는 %s 입니다." %a
    else:
        return "오늘의 발표자는 %s 입니다." %b


# In[ ]:

#print(vote('이현우','조아영'))


