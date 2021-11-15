#!/usr/bin/env python
# coding: utf-8

# # <u>Take Care of Your Employees</u>

# Employees
# Take care of your employees: A drastic thing happened and ABCXYZ123 Technical Solutions have lost one of their important employees. The company is now very concerned about the health of their employees and would want you to find that set of employees who are in need or may be in need of treatment, taking into account multiple attributes that are already stored in the database. So buckle up the wellness of your employees is in your hand.

# <u>The objective of the problem:</u>
# 
# The objective is to predict values “treatment” attribute from the given features of the Test data.
# 
# The predictions are to be written to a CSV file along with ID which is the unique identifier for each tuple.
# Please upload the submission file to get a score.
# Please note that the training data is only for creating your data model and all predictions are to be made as per serial numbers on the test file.

# <u>The description of the data attributes is as below:</u>
# 
#     • Timestamp
#     • Age
#     • Gender
#     • Country
#     • state: If you live in the United States, which state or territory do you live in?
#     • self_employed: Are you self-employed?
#     • family_history: Do you have a family history of mental illness?
#     • treatment: Does he or she really needs treatment.
#     • work_interfere: If you have a mental health condition, do you feel that it interferes with your work?
#     • no_employees: How many employees does your company or organization have?
#     • remote_work: Do you work remotely (outside of an office) at least 50% of the time?
#     • tech_company: Is your employer primarily a tech company/organization?
#     • benefits: Does your employer provide mental health benefits?
#     • care_options: Do you know the options for mental health care your employer provides?
#     • wellness_program: Has your employer ever discussed mental health as part of an employee wellness program?
#     • seek_help: Does your employer provide resources to learn more about mental health issues and how to seek help?
#     • anonymity: Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?
#     • leave: How easy is it for you to take medical leave for a mental health condition?
#     • mental_health_consequence: Do you think that discussing a mental health issue with your employer would have negative consequences?
#     • phys_health_consequence: Do you think that discussing a physical health issue with your employer would have negative consequences?
#     • coworkers: Would you be willing to discuss a mental health issue with your coworkers?
#     • supervisor: Would you be willing to discuss a mental health issue with your direct supervisor(s)?
#     • mental_health_interview: Would you bring up a mental health issue with a potential employer in an interview?
#     • phys_health_interview: Would you bring up a physical health issue with a potential employer in an interview?
#     • mental_vs_physical: Do you feel that your employer takes mental health as seriously as physical health?
#     • obs_consequence: Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?
#     • comments: Any additional notes or comments.

# <u>Load Basic Libraries</u>

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# To display maximum columns
pd.set_option("display.max_columns",None)


# <u>Load the csv files into dataframe</u>

# In[2]:


train_df = pd.read_csv(r'training_.csv', header = 0, index_col = 0)
test_df = pd.read_csv(r'test.csv', header = 0, index_col = 0)

# creating a copy of original df
train_copy = train_df.copy()
test_copy = test_df.copy()


# In[3]:


print('Shape : ',train_df.shape)
train_df.head()


# In[4]:


print('Shape : ',test_df.shape)
test_df.head()


# ### Pre-processing the data

# <u>Let us check the data frame description</u>

# In[5]:


train_df.describe(include = 'all')


# In[6]:


test_df.describe(include = 'all')


# In[7]:


train_df.info()


# In[8]:


test_df.info()


# - From above we do see that we have missing values in both the data frames. Let us check for missing values

# <u>Checking for null values</u>

# In[9]:


print('Training Data : \n')
train_df.isnull().sum()


# In[10]:


print('Testing Data : \n')
test_df.isnull().sum()


# ##### <u>Training Data :</u>
#         -In this dataset we see that 'state' , 'self_employed', 'work_interfere' & 'comments' are the columns having missing values.
# ##### <u>Testing Data :</u>
#         -In this dataset we see that 'state', 'work_interfere' & 'comments' are the columns having missing values.
#         
#         
# - As our problem statement is predicting whether 'treatment' is required or not, we see that 'comments' will not play an important role as it basically explains the employee's choices of answering and is not helpoing on the target variable      
# - The state column (state: If you live in the United States, which state or territory do you live in?) basically is applicable for USA and not other countries. So as part of feature selection we shall drop the column
# - As per above statements, we only need to handle missing values for 'self_employed' & 'work_interfere' in this dataset

# In[11]:


train_df['work_interfere'].unique()


# In[12]:


train_df['self_employed'].unique()


# - We see that work_interface is an ordinal data having categories. So we shall handle the null values by adding a category called Dont know similar to what we see in leave column
# - As self_employed is a binary categorical data we shall replace it with the MODE value

# In[13]:


print('self_employed => trainData:: will be replaced by Mode and value is ',train_df['self_employed'].mode()[0])
print('work_interfere => trainData:: will be replaced by Mode and value is ',train_df['work_interfere'].mode()[0])
train_df['self_employed'].fillna(train_df['self_employed'].mode()[0], inplace = True)
train_df['work_interfere'].fillna("Don't know", inplace = True)


# In[14]:


test_df['self_employed'].fillna(test_df['self_employed'].mode()[0], inplace = True)
test_df['work_interfere'].fillna("Don't know", inplace = True)


# In[15]:


print('Trainng dataset :\n',train_df.isnull().sum())
print('\nTesting dataset : \n',test_df.isnull().sum())


# - We have imputed the desired columns

# - now before we move ahead, let us drop the columns as decided in the above step

# In[16]:


train_df.drop(['Timestamp','Country', 'state', 'comments'], axis = 1, inplace = True)
test_df.drop(['Timestamp', 'Country','state', 'comments'], axis = 1, inplace = True)
print('Training dataset after droping columns -',train_df.shape)
print('Testing dataset after droping columns -',test_df.shape)


# In[17]:


train_df.head()


# In[18]:


test_df.head()


# - Now Let us check if the data in the datasets have proper categorical values, so that we can encode them later during model building

# In[19]:


def print_unique_values(df):
    for colname in df.columns:
        if df[colname].dtype == 'object' or df[colname].dtype == 'int64':
            print('--------- {} -----------'.format(colname))
            print(np.sort(df[colname].unique()),'\nCount ::', df[colname].nunique(),'\n')


# In[20]:


print_unique_values(train_df)


#     - --------- Gender -----------
#     ['A little about you' 'Agender' 'All' 'Androgyne' 'Cis Female' 'Cis Male'
#      'Enby' 'F' 'Femake' 'Female' 'Female ' 'Female (cis)' 'Female (trans)'
#      'Genderqueer' 'Guy (-ish) ^_^' 'M' 'Mail' 'Make' 'Mal' 'Male' 'Male '
#      'Male (CIS)' 'Male-ish' 'Malr' 'Man' 'Nah' 'Neuter' 'Trans woman'
#      'Trans-female' 'Woman' 'cis male' 'cis-female/femme' 'f' 'female' 'fluid'
#      'm' 'maile' 'male' 'male leaning androgynous' 'msle' 'non-binary' 'queer'
#      'queer/she/they' 'something kinda male?' 'woman'] 
#     Count :: 45 
#         -
#         - As we see There is a data mismatch in the Gender column.
#         - We shall replace all values of male by m, Female by f and other data as o representing others

#     --------- Age -----------
#     [      -1726         -29           5           8          18          19
#               20          21          22          23          24          25
#               26          27          28          29          30          31
#               32          33          34          35          36          37
#               38          39          40          41          42          43
#               44          45          46          47          48          49
#               50          51          53          54          55          56
#               57          58          60          61          62          65
#              329 99999999999] 
#     Count :: 50  
# 
#         - We also see that age has negative values and values above 100 as well for an employee
#    [Country wise retirement age](https://en.wikipedia.org/wiki/Retirement_age)
#         - As per the wiki link we see that the 65 is the upper limit of employee working. So we shall filter out this column  and Impute the courrupted values 

# In[21]:


print_unique_values(test_df)


# - In the testing data we see the same as we saw in the training data for Age and Gender

# ---------------------------------------------------------------------------------    
#     Replacing the Traing data for Gender
#     
#     ['A little about you' 'Agender' 'All' 'Androgyne' 'Cis Female' 'Cis Male'
#          'Enby' 'F' 'Femake' 'Female' 'Female ' 'Female (cis)' 'Female (trans)'
#          'Genderqueer' 'Guy (-ish) ^_^' 'M' 'Mail' 'Make' 'Mal' 'Male' 'Male '
#          'Male (CIS)' 'Male-ish' 'Malr' 'Man' 'Nah' 'Neuter' 'Trans woman'
#          'Trans-female' 'Woman' 'cis male' 'cis-female/femme' 'f' 'female' 'fluid'
#          'm' 'maile' 'male' 'male leaning androgynous' 'msle' 'non-binary' 'queer'
#          'queer/she/they' 'something kinda male?' 'woman'] 
#          
#          
#  ---------------------------------------------------------------------------------    
#     Replacing the Test data for Gender
#     
#     ['Cis Man' 'F' 'Female' 'M' 'Male' 'Male ' 'Woman' 'f' 'femail' 'female' 'm' 'male' 'ostensibly male, unsure what that really means' 'p'] 

# In[22]:


# Female :: F
train_df['Gender'] = train_df['Gender'].replace(['Cis Female','F', 'Femake', 'Female', 'Female ', 'Female (cis)','Woman', 'cis-female/femme', 'f', 'female', 'woman', 'femail'],'F')
test_df['Gender'] = test_df['Gender'].replace(['F', 'Female','Woman', 'f', 'femail', 'female'], 'F')

# Male :: M
train_df['Gender'] = train_df['Gender'].replace(['Cis Male', 'M', 'Mail', 'Make', 'Mal', 'Male', 'Male ','Male (CIS)', 'Male-ish', 'Malr', 'Man','cis male', 'Cis Man', 'm', 'maile', 'male', 'msle', 'something kinda male?','ostensibly male, unsure what that really means'],'M')
test_df['Gender'] = test_df['Gender'].replace(['Cis Man', 'M', 'Male', 'Male ', 'm', 'male', 'ostensibly male, unsure what that really means'], 'M')

#Others :: O
train_df['Gender'] = train_df['Gender'].replace(['A little about you', 'Agender' ,'All', 'Androgyne','Enby','Female (trans)', 'Genderqueer', 'p', 'Guy (-ish) ^_^', 'Nah', 'Neuter', 'Trans woman', 'Trans-female', 'fluid', 'male leaning androgynous', 'non-binary', 'queer', 'queer/she/they'],'O')
test_df['Gender'] = test_df['Gender'].replace(['p'], 'O')


# In[23]:


print('Unique value for Gender in Training :',train_df['Gender'].unique())
print(train_df['Gender'].value_counts())
print('\nUnique value for Gender in Testing  :',test_df['Gender'].unique())
print(test_df['Gender'].value_counts())


# --------------------------------------------------------------------
# Working on the Age part

#     Training data
#     --------- Age -----------
#     [      -1726         -29           5           8          18          19
#               20          21          22          23          24          25
#               26          27          28          29          30          31
#               32          33          34          35          36          37
#               38          39          40          41          42          43
#               44          45          46          47          48          49
#               50          51          53          54          55          56
#               57          58          60          61          62          65
#              329 99999999999] 
#     Count :: 50  
# 
#         - We also see that age has negative values and values above 100 as well for an employee
#    [Country wise retirement age](https://en.wikipedia.org/wiki/Retirement_age)
#         - As per the wiki link we see that the 65 is the upper limit of employee working. So we shall filter out this column  and Impute the courrupted values 
#             
#         testing data
#         --------- Age -----------
#     [-1 11 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41
#      42 43 44 45 46 48 50 51 56 60 72] 

# In[24]:


# Employment age is from 18 to 65
train_age_mean = int(train_df[(train_df['Age'] <= 65) & (train_df['Age'] >= 18)].Age.mean())
test_age_mean = int(test_df[(test_df['Age'] <= 65) & (test_df['Age'] >= 18)].Age.mean())


# In[25]:


train_df['Age'] = train_df['Age'].apply(lambda x: train_age_mean if (x < 18 or x > 65) else x)
test_df['Age'] = test_df['Age'].apply(lambda x: test_age_mean if (x < 18 or x > 65) else x)


# - We have replaced the age value which were not in significant range by the mean values

# In[26]:


train_df.Age.describe()


# In[27]:


test_df.Age.describe()


# - We see that we have successfully imputed the values by looking at the min and max value

# <u>no_employee</u>'
# 
#         - We see that no_employees have a range and hence we shall split the range into two columns as lower and upper for better prediction of the model
#         - Also we shall provide an upper limit for the value 'More than 1000'
#         --------- no_employees -----------
#         ['1-5' '100-500' '26-100' '500-1000' '6-25' 'More than 1000'] 
#         Count :: 6 

# - We also understand that Timestamp will be of no significance in the prediction. So we drop the variable before we start encoding the data

# #### <u>Encoding the train and test data for model building</u>

# - We shall encode the categorical data using labelEncoder

# In[28]:


# separating the target variable from train data
train_df_Y = train_df['treatment']
train_df.drop(['treatment'], axis = 1, inplace = True)


# In[29]:


train_df['treatment'] = train_df_Y


# In[30]:


# Encding the data using Label encoder


# In[31]:


# For preprocessing the data
from sklearn.preprocessing import LabelEncoder

encoder_label = LabelEncoder()

for col_name in test_df.columns:
    if train_df[col_name].dtype == 'object':
        train_df[col_name] = encoder_label.fit_transform(train_df[col_name])
        test_df[col_name] = encoder_label.fit_transform(test_df[col_name])
        
train_df['treatment'] = encoder_label.fit_transform(train_df['treatment'])


# In[32]:


train_df.head()


# In[33]:


test_df.head()


# ### <u>Splitting the training data into X and Y</u>

# In[34]:


X = train_df.iloc[:,0:-1]
Y = train_df.iloc[:,-1]


# In[35]:


Y.value_counts()


# ### <u>Scaling the Data</u>

# In[36]:


X.head()


# In[37]:


# Scaling the data
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
scaler.fit(X)
X= scaler.transform(X)


# In[38]:


test_df.head()


# In[39]:


scaler.fit(test_df)
test_df = scaler.transform(test_df)


# In[40]:


# Splitting the data in 70:30
from sklearn.model_selection import train_test_split
#Split the data into test and train using stratified K-Fold
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,random_state=10, stratify = Y)  


# In[41]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
#Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[42]:


def evaluation(Y_test,Y_pred, plot_prediction):
    

    cfm=confusion_matrix(Y_test,Y_pred)
    print("Confusion Metrics :\n",cfm)

    print("\n")
    print(pd.crosstab(Y_test, Y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    
    print("\nClassification report: \n")

    print(classification_report(Y_test,Y_pred))

    acc=accuracy_score(Y_test, Y_pred) #
    print("\nAccuracy of the model: ",acc)
    print('\n----- YTest v Y pred (30 values)---------')
    print('YTest :',Y_test.values[0:31])
    print('YPred :',Y_pred[0:31])
    
    if plot_prediction :
        plt.figure(figsize=(25,6))
        plt.title('Y_test v Y_pred for first 40 values')
        plt.plot(Y_test.values[0:41], label='Actual', linestyle='--', marker='o', color='g')
        plt.plot(Y_pred[0:41], label='Predicted', linestyle='--', marker='o', color='r')
        plt.legend(prop={'size': 20})
        plt.show()


# ### Logistic Regression

# In[43]:


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
evaluation(Y_test,Y_pred, True)


# ### KNN

# In[44]:


neighbors = np.arange(1,41)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
my_dict = {}

for i,k in enumerate(neighbors):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k, metric="manhattan")
    
    #Fit the model
    knn.fit(X_train, Y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, Y_train)
    
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, Y_test)
    
    #Predict 
    Y_pred = knn.predict(X_test)
    
    # Zipping Accuracy score
    my_dict[k]=accuracy_score(Y_test,Y_pred)
    
print(my_dict)


# In[45]:


knn = KNeighborsClassifier(n_neighbors = 37, metric = "manhattan")                                
knn.fit(X_train,Y_train)                               
Y_pred = knn.predict(X_test)
evaluation(Y_test,Y_pred, False)


# ### Decision Tree

# In[46]:


decisionTree = DecisionTreeClassifier(criterion = "gini", random_state=10, max_depth=3)
decisionTree.fit(X_train, Y_train)
Y_pred = decisionTree.predict(X_test)
evaluation(Y_test,Y_pred, False)


# In[47]:


# Boosting the decision tree


# In[48]:


#predicting using the AdaBoost_Classifier
from sklearn.ensemble import AdaBoostClassifier
model_AdaBoost=AdaBoostClassifier(base_estimator=decisionTree,n_estimators=2,random_state=10)
#fit the model on the data and predict the values
model_AdaBoost.fit(X_train,Y_train)
Y_pred=model_AdaBoost.predict(X_test)
evaluation(Y_test,Y_pred, False)


# ### RandomForestClassifier

# In[49]:


random_tree=RandomForestClassifier(n_estimators=100, random_state=10, max_depth = 4)
random_tree.fit(X_train,Y_train)
Y_pred=random_tree.predict(X_test)
evaluation(Y_test,Y_pred, False)


# ### SVC

# In[50]:


model_SVC=svm.SVC(kernel="rbf", gamma=0.1, C=0.5)
model_SVC.fit(X_train,Y_train)
Y_pred=model_SVC.predict(X_test)
evaluation(Y_test,Y_pred, False)


# ### ExtraTreesClassifier

# In[51]:


model_EXT=ExtraTreesClassifier(n_estimators=250, random_state=10)
model_EXT.fit(X_train,Y_train)
Y_pred=model_EXT.predict(X_test)
evaluation(Y_test,Y_pred, False)


# ### GradientBoostingClassifier

# In[52]:


model_GradientBoosting=GradientBoostingClassifier(n_estimators=250,random_state=10)
model_GradientBoosting.fit(X_train,Y_train)
Y_pred=model_GradientBoosting.predict(X_test)
evaluation(Y_test,Y_pred, False)


# In[53]:


from sklearn.ensemble import VotingClassifier
# create the sub models
estimators = []
#model1 = LogisticRegression()
#estimators.append(('log', model1))
model2 = DecisionTreeClassifier(criterion = "gini", random_state=10, max_depth=4)
estimators.append(('cart', model2))
model3 = svm.SVC(kernel="rbf", C=1,gamma=0.1)
estimators.append(('svm', model3))
model4 = KNeighborsClassifier(n_neighbors=37, metric='manhattan')
estimators.append(('knn', model4))
model5 = GradientBoostingClassifier(n_estimators=150, random_state=10)
estimators.append(('GradientBoostingClassifier', model5))

# create the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(X_train,Y_train)
Y_pred=ensemble.predict(X_test)
evaluation(Y_test,Y_pred,False)


# In[54]:


Y1 = random_tree.predict(test_df)
Y2 = decisionTree.predict(test_df)
Y3 = model_GradientBoosting.predict(test_df)
Y4 = logreg.predict(test_df)


# #### we shall be going ahead with Decision tree as it has the highest accuracy in class1 prediction

# In[58]:


df=pd.DataFrame()
df['S.No '] = np.arange(1,(len(test_df)+1))
df['treatment'] = Y2
df['treatment'] = df['treatment'].map({1:'Yes', 0:'No'})

df.to_csv("outcome.csv",header=True,index=False)


# In[60]:


# Personal Testing


# In[59]:


new_df=pd.DataFrame(index=pd.RangeIndex(start=1, stop=len(test_df)+1, name='S.No '))
new_df["Random"]=Y1
new_df['Decision']=Y2
new_df['gradient']=Y3
new_df['linear']=Y4

new_df['Decision'] = new_df['Decision'].map({1:'Yes', 0:'No'})
new_df["Random"] = new_df["Random"].map({1:'Yes', 0:'No'})
new_df['gradient']=new_df["gradient"].map({1:'Yes', 0:'No'})
new_df['linear']=new_df["linear"].map({1:'Yes', 0:'No'})

new_df.to_excel("Pre.xlsx",header=True,index=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




