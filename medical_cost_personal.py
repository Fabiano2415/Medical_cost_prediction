#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd


# In[51]:


df = pd.read_csv("insurance.csv")
# the first 5 rows
df.head()


# In[52]:


# The last 5 rows
df.tail()


# In[53]:


#shape
df.shape


# In[54]:


print("Number of Rows:",df.shape[0])
print("Number of Columns:",df.shape[1])


# In[55]:


df.info()


# In[56]:


# no missing or null values
df.isnull().sum()


# In[57]:


df.describe()


# In[58]:


df.head()


# ####  Covert Columns From String ['sex' ,'smoker','region' ] To Numerical Values

# In[59]:


df['sex'].unique()


# In[60]:


# convert female to 0 and male to 1
df['sex'] = df['sex'].map({'female':0,'male':1})


# In[61]:


#convert smoker to 1 and no smoker to 0
df['smoker']= df['smoker'].map({'yes':1,'no':0})


# In[62]:


df['region'].unique()


# In[63]:


df['region']=df['region'].map({'southwest':1,'southeast':2,
                   'northwest':3,'northeast':4})


# In[64]:


#df['region']=df['region'].map({'southwest':1,'southeast':2,
#                    'northwest':3,'northeast':4})


# In[65]:


df.head()


# #### Store Feature Matrix In X and Response(Target) In Vector y

# In[67]:


df.columns


# In[68]:


X = df.drop(['charges'],axis=1)


# In[69]:


y = df['charges']


# ##### - Train/Test split

# In[70]:


from sklearn.model_selection import train_test_split


# In[71]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[72]:


y_train


# In[73]:


X_train


# In[74]:


# Import the models
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[75]:


#model training
lr = LinearRegression()
lr.fit(X_train,y_train)
svm = SVR()
svm.fit(X_train,y_train)
rf = RandomForestRegressor()
rf.fit(X_train,y_train)
gr = GradientBoostingRegressor()
gr.fit(X_train,y_train)


# In[77]:


# Prediction on Test Data

y_pred1 = lr.predict(X_test)
y_pred2 = svm.predict(X_test)
y_pred3 = rf.predict(X_test)
y_pred4 = gr.predict(X_test)

df1 = pd.DataFrame({'Actual':y_test,'Lr':y_pred1,
                  'svm':y_pred2,'rf':y_pred3,'gr':y_pred4})


# In[78]:


df1


# In[79]:


#Evaluate the Models
from sklearn import metrics


# In[80]:


score1 = metrics.r2_score(y_test,y_pred1)
score2 = metrics.r2_score(y_test,y_pred2)
score3 = metrics.r2_score(y_test,y_pred3)
score4 = metrics.r2_score(y_test,y_pred4)


# In[81]:


print(score1,score2,score3,score4)


# In[82]:


s1 = metrics.mean_absolute_error(y_test,y_pred1)
s2 = metrics.mean_absolute_error(y_test,y_pred2)
s3 = metrics.mean_absolute_error(y_test,y_pred3)
s4 = metrics.mean_absolute_error(y_test,y_pred4)


# print(s1,s2,s3,s4)

# #### Cost estimates for new customers

# In[84]:


data = {'age' : 40,
        'sex' : 1,
        'bmi' : 40.30,
        'children' : 4,
        'smoker' : 1,
        'region' : 2}

df = pd.DataFrame(data,index=[0])
df


# In[85]:


new_pred = gr.predict(df)
print("Medical Insurance cost for new : ",new_pred)

# Save Model Usign Joblib
# In[86]:


import joblib


# In[87]:


joblib.dump(gr,'model_joblib_test')


# In[88]:


model = joblib.load('model_joblib_test')


# In[89]:


model.predict([[40,1,40.3,4,1,2]])


# #### GUI

# In[90]:


from tkinter import *


# In[91]:


import joblib


# In[92]:


def show_entry():
    
    p1 = float(e1.get())
    p2 = float(e2.get())
    p3 = float(e3.get())
    p4 = float(e4.get())
    p5 = float(e5.get())
    p6 = float(e6.get())

    model = joblib.load('model_joblib_gr')
    result = model.predict([[p1,p2,p3,p4,p5,p6]])
    
    Label(master, text = "Insurance Cost").grid(row=7)
    Label(master, text=result).grid(row=8)


master =Tk()
master.title("Insurance Cost Prediction")
label = Label(master,text = "Insurance Cost Prediction",bg = "black",
               fg = "white").grid(row=0,columnspan=2)

Label(master,text = "Enter Your Age").grid(row=1)
Label(master,text = "Male Or Female [1/0]").grid(row=2)
Label(master,text = "Enter Your BMI Value").grid(row=3)
Label(master,text = "Enter Number of Children").grid(row=4)
Label(master,text = "Smoker Yes/No [1/0]").grid(row=5)
Label(master,text = "Region [1-4]").grid(row=6)


e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)



e1.grid(row=1,column=1)
e2.grid(row=2,column=1)
e3.grid(row=3,column=1)
e4.grid(row=4,column=1)
e5.grid(row=5,column=1)
e6.grid(row=6,column=1)


Button(master,text="Predict",command=show_entry).grid()

mainloop()


# In[ ]:




