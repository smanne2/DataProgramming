
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


file_input = "cancerData.csv"


# In[3]:


file = pd.read_csv(file_input)


# In[4]:


import numpy as np
np.unique(file.Class)


# In[5]:


list(enumerate(np.unique(file.Class)))


# In[6]:


class_mapping = {label: idx for idx, label in enumerate(np.unique(file.Class))}
print(class_mapping)


# In[7]:


file['Class'] = file['Class'].map(class_mapping)
print(file)


# In[8]:


from sklearn.preprocessing import MinMaxScaler


# In[9]:


scaler = MinMaxScaler(feature_range=(0,2))
new_df = scaler.fit_transform(file)
print(new_df)


# In[10]:


norm_df = pd.DataFrame(data=new_df[:,:], columns=file.columns)
print(norm_df)


# In[11]:


# norm_df = (file - file.min())/(file.max() - file.min())
# print(norm_df.head())


# In[13]:


norm_df.to_csv("cancerNormalized.csv",index=False)

