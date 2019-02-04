#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))


# In[ ]:


dataset = pd.read_csv("../input/train.csv")
testset = pd.read_csv("../input/test.csv")


# In[ ]:


dataset.head()


# In[ ]:


dataset.info()


# In[ ]:


label = dataset.iloc[:,1]
info = dataset.iloc[:,[2,4,5]]
testdata = testset.iloc[:,[1,3,4]]

x = [info,testdata]

for i in x:
    i['Sex'] = i['Sex'].map({'female':0,'male': 1}).astype(float)
    
info = info.fillna(0).astype(float)
testdata = testdata.fillna(0).astype(float)


# In[ ]:


from sklearn.model_selection import train_test_split

train_data,test_data,train_labels,test_labels=train_test_split(info,label,random_state=7,train_size=0.7)


# In[ ]:


from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(info,label)


# In[ ]:


predictions = clf.predict(test_data)
from sklearn.metrics import accuracy_score
print(accuracy_score(test_labels,predictions))


# In[ ]:


result = clf.predict(testdata)
result=np.array(result,dtype='int')
print(result)   


# In[ ]:


indice = [testset['PassengerId']]
df=pd.DataFrame(data=result,index=testset['PassengerId'],columns=['Survived'])
df.to_csv('gender_submission.csv',header=True)
print('gender_submission.csv')

