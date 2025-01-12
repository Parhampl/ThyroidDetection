#!/usr/bin/env python
# coding: utf-8

# ## Import necessary libraries
# 

# In[60]:

##Parham_Porkhial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# ## Load the dataset
# 

# In[73]:


dataset = pd.read_csv('ThyroidDetection.csv')

print(dataset.head())


# ## Preprocessing
# 

# In[63]:


X = dataset.drop('Thyroid Status', axis=1)
y = dataset['Thyroid Status']


# ## Convert non-numeric columns to numeric using Label Encoding
# 

# In[64]:



from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
X_encoded = X.apply(label_encoder.fit_transform)


# ## Split the dataset into training and testing sets
# 

# In[65]:



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# ## Implementing RandomForest
# 

# In[66]:



from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)


# ## Train the model
# 

# In[67]:


model.fit(X_train, y_train)


# ## Make predictions on the test set
# 

# In[68]:


y_pred = model.predict(X_test)


# ## Evaluate the model
# 

# In[69]:


from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
classification_report_dict = classification_report(y_test, y_pred, output_dict=True)


# ## Print overall accuracy
# 

# In[70]:


print(f'Overall Accuracy: {accuracy:.4f}\n')


# ## Print the results for each class
# 

# In[72]:


for class_label, metrics in classification_report_dict.items():
    
    if not any(char.isdigit() for char in class_label):
        continue

    class_index = int(float(class_label))
    print(f"Class {class_index}:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")  #(2*recall*percision)/(percission+recall)
    print(f"F1-Score: {metrics['f1-score']:.4f}")
    print(f"Support: {metrics['support']}")  #number of actual occurrences of the class in the dataset
    print()


# In[75]:


pip install pypandoc

