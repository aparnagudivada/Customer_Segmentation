#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np


# In[4]:


import pandas as pd


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


import seaborn as sns


# In[16]:


df = pd.read_csv("Mall_customers.csv")


# In[17]:


df.head()


# In[11]:


df.drop(["CustomerID"],axis=1,inplace=True)


# In[12]:


plt.figure(figsize=(6,6))
plt.title("Ages Frequency")
sns.axes_style("ticks")
sns.violinplot(y=df["Age"],color="red")
plt.show()


# In[13]:


plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.boxplot(y=df["Spending Score (1-100)"], color="pink")
plt.subplot(1,2,2)
sns.boxplot(y=df["Annual Income (k$)"],color="yellow")
plt.show()


# In[18]:


genders = df.Genre.value_counts()
sns.set_style("dark")
plt.figure(figsize=(10,4))
sns.barplot(x=genders.index, y=genders.values,color="green")
plt.show()


# In[19]:


age18_25 = df.Age[(df.Age <= 25) & (df.Age >= 18)]
age26_35 = df.Age[(df.Age <= 35) & (df.Age >= 26)]
age36_45 = df.Age[(df.Age <= 45) & (df.Age >= 36)]
age46_55 = df.Age[(df.Age <= 55) & (df.Age >= 46)]
age55above = df.Age[df.Age >= 56]

x = ["18-25","26-35","36-45","46-55","55+"]
y = [len(age18_25.values),len(age26_35.values),len(age36_45.values),len(age46_55.values),len(age55above.values)]

plt.figure(figsize=(15,6))
sns.barplot(x=x, y=y, palette="winter")
plt.title("Number of Customer and Ages")
plt.xlabel("Age")
plt.ylabel("Number of Customer")
plt.show()


# In[20]:


ss1_20 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 1) & (df["Spending Score (1-100)"] <= 20)]
ss21_40 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 21) & (df["Spending Score (1-100)"] <= 40)]
ss41_60 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 41) & (df["Spending Score (1-100)"] <= 60)]
ss61_80 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 61) & (df["Spending Score (1-100)"] <= 80)]
ss81_100 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 81) & (df["Spending Score (1-100)"] <= 100)]

ssx = ["1-20", "21-40", "41-60", "61-80", "81-100"]
ssy = [len(ss1_20.values), len(ss21_40.values), len(ss41_60.values), len(ss61_80.values), len(ss81_100.values)]

plt.figure(figsize=(15,6))
sns.barplot(x=ssx, y=ssy, palette="nipy_spectral_r")
plt.title("Spending Scores")
plt.xlabel("Score")
plt.ylabel("Number of Customer Having the Score")
plt.show()


# In[21]:


ai0_30 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 0) & (df["Annual Income (k$)"] <= 30)]
ai31_60 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 31) & (df["Annual Income (k$)"] <= 60)]
ai61_90 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 61) & (df["Annual Income (k$)"] <= 90)]
ai91_120 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 91) & (df["Annual Income (k$)"] <= 120)]
ai121_150 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 121) & (df["Annual Income (k$)"] <= 150)]

aix = ["$ 0 - 30,000", "$ 30,001 - 60,000", "$ 60,001 - 90,000", "$ 90,001 - 120,000", "$ 120,001 - 150,000"]
aiy = [len(ai0_30.values), len(ai31_60.values), len(ai61_90.values), len(ai91_120.values), len(ai121_150.values)]

plt.figure(figsize=(15,6))
sns.barplot(x=aix, y=aiy, palette="spring")
plt.title("Annual Incomes")
plt.xlabel("Income")
plt.ylabel("Number of Customer")
plt.show()


# In[22]:


from mpl_toolkits.mplot3d import Axes3D

sns.set_style("white")
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.Age, df["Annual Income (k$)"], df["Spending Score (1-100)"], c='red', s=60)
ax.view_init(30, 185)
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel('Spending Score (1-100)')
plt.show()


# In[23]:


get_ipython().system('pip3 install scikit-learn')


# In[30]:


df.iloc[:,1:]


# In[33]:



from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder


# In[34]:


encoder = OneHotEncoder()
genre_encoded = encoder.fit_transform(df[['Genre']]).toarray()
genre_df = pd.DataFrame(genre_encoded, columns=encoder.get_feature_names_out(['Genre']))

# Combine the numerical and one-hot encoded columns
df_combined = pd.concat([df.drop(['CustomerID', 'Genre'], axis=1), genre_df], axis=1)

# Now apply KMeans
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(df_combined)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(12, 6))
plt.grid()
plt.plot(range(1, 11), wcss, linewidth=2, color="blue", marker="8")
plt.xlabel("K Value")
plt.xticks(np.arange(1, 11, 1))
plt.ylabel("WCSS")
plt.show()


# In[38]:


km = KMeans(n_clusters=5)
clusters = km.fit_predict(df_combined)

df["label"] = clusters

# 3D scatter plot after clustering
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')

colors = ['blue', 'red', 'green', 'orange', 'purple']
for i in range(5):
    ax.scatter(df.Age[df.label == i], 
               df["Annual Income (k$)"][df.label == i], 
               df["Spending Score (1-100)"][df.label == i], 
               c=colors[i], s=60, label=f'Cluster {i}')

ax.view_init(30, 185)
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel('Spending Score (1-100)')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




