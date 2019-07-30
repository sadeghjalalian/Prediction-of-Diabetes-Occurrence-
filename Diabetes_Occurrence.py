#!/usr/bin/env python
# coding: utf-8

# In[84]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import metrics


# In[86]:


#importing the dataset
df = pd.read_csv('/Users/sadegh/Desktop/DataSet GitHub/Decision Tree/pima_native_american_diabetes_weka_dataset.csv')
df.head(7)


# In[39]:


df.info()


# In[40]:


plt.figure(figsize=(12,8))
sns.heatmap(df.describe()[1:].transpose(),
            annot=True,linecolor="w",
            linewidth=2,cmap=sns.color_palette("Set1"))
plt.title("Data summary")
plt.show()


# In[41]:


correlation = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation,annot=True,
            cmap=sns.color_palette("magma"),
            linewidth=2,edgecolor="k")
plt.title("CORRELATION BETWEEN VARIABLES")
plt.show()


# In[53]:


plt.figure(figsize=(12,6))
plt.pie(df["class"].value_counts().values,
        labels=["no diabets","diabets"],
        autopct="%1.0f%%",wedgeprops={"linewidth":2,"edgecolor":"white"})
my_circ = plt.Circle((0,0),.7,color = "white")
plt.gca().add_artist(my_circ)
plt.subplots_adjust(wspace = .2)
plt.title("Proportion of target variable in dataset")
plt.show()


# In[93]:


plt.figure(figsize=(12,6))
sns.scatterplot(data=df,x='age',y='times_pregnant',hue='class',cmap="Set2")
plt.legend(title='legend',loc='upper right', labels=['no diabets', 'diabets'])


# In[77]:


df[(df['class'] ==1)].mean().reset_index()


# In[43]:


X = df.iloc[:,:-1]
Y = df.iloc[:,8]


# In[44]:


#Splitting the data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.25, random_state = 0)


# In[45]:


#fitting classifier to the training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state = 0)
classifier.fit(X_train,y_train)


# In[46]:


y_pred = classifier.predict(X_test)


# In[47]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


# In[48]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="BuPu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[49]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[50]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[55]:


from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 

features = list(df.columns[:-1])
features


# In[95]:


plt.figure(figsize=(14,8))
dot_data = StringIO()  
export_graphviz(classifier, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())  


# In[ ]:




