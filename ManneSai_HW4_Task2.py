
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


file_input = "cancerNormalized.csv"


# In[3]:


df = pd.read_csv(file_input)


# In[4]:


import numpy as np
from sklearn.model_selection import train_test_split


# In[5]:


X, y = df.iloc[:,1:], df['Class']


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = (1/3), random_state = 0, stratify = y)


# In[7]:


print(X_train.head())


# In[8]:


print(y_train.head())


# In[9]:


df_tr_X = pd.DataFrame(X_train)
df_tr_y = pd.DataFrame(y_train)
df_t_X = pd.DataFrame(X_test)
df_t_y = pd.DataFrame(y_test)


# In[10]:


df_tr_X.to_csv("X_train.csv", index = False)
df_tr_y.to_csv("y_train.csv", index = False)
df_t_X.to_csv("X_test.csv", index = False)
df_t_y.to_csv("y_test.csv", index = False)


# In[11]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz


# In[12]:


import graphviz


# In[13]:


result = pd.DataFrame(columns = ['LevelLimit', 'Score for Training', 'Score for Testing'])
for treeDepth in range(1,11):
    dct = DecisionTreeClassifier(criterion='entropy', max_depth=treeDepth, random_state=0)
    dct = dct.fit(X_train, y_train)
    dct.predict(X_test)
    scoreTrain = dct.score(X_train, y_train)
    scoreTest = dct.score(X_test, y_test)
    result.loc[treeDepth] = [treeDepth,scoreTrain, scoreTest]
print(result.head(11))


# In[14]:


from matplotlib import pyplot
ax = pyplot.plot(result.LevelLimit, result['Score for Training'], result['Score for Testing'])
ay = pyplot.title("Graph for Entropy Decision Tree Classifier")
a=pyplot.xlabel("Tree Depth")
b=pyplot.ylabel("Score")
c=pyplot.legend()


# In[15]:


path = pd.ExcelWriter("Results.xlsx")
result.to_excel(path, 'Task2_Entropy', index=False)
# path.save()
# path.close()


# In[16]:


result2 = pd.DataFrame(columns = ['LevelLimit', 'Score for Training', 'Score for Testing'])
for treeDepth in range(1,11):
    dct_gini = DecisionTreeClassifier(criterion='gini', max_depth=treeDepth, random_state=0)
    dct_gini = dct_gini.fit(X_train, y_train)
    dct_gini.predict(X_test)
    scoreTrain = dct_gini.score(X_train, y_train)
    scoreTest = dct_gini.score(X_test, y_test)
    result2.loc[treeDepth] = [treeDepth,scoreTrain, scoreTest]
print(result2.head(11))


# In[17]:


from matplotlib import pyplot
ax = pyplot.plot(result2.LevelLimit, result2['Score for Training'], result2['Score for Testing'])
ay = pyplot.title("Graph for Gini Decision Tree Classifier")
a=pyplot.xlabel("Tree Depth")
b=pyplot.ylabel("Score")
c=pyplot.legend()


# In[18]:


result2.to_excel(path, 'Task2_Gini', index = False)
path.save()
path.close()


# In[ ]:


dot_data=export_graphviz(dct_gini, out_file = None, rounded=True, special_characters=True, filled=True)
graph = graphviz.Source(dot_data)
graph.render("Tree")

