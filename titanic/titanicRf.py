import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from pandas.api.types import is_numeric_dtype
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
from statsmodels.formula.api import ols

pLE=preprocessing.LabelEncoder()
# import data
df=pd.read_csv('../input/titanic/train.csv')
test=pd.read_csv("../input/titanic/test.csv")

# mske new data columns
def procesDat(data):
    data['NameL']=0
    data['FamGr']=0
    for i in range(0, len(data.index)):
        data['NameL'][i]=len(data['Name'][i])
    data['Cabin'].fillna('Uknown', inplace=True)
    data['Cabin'] = data['Cabin'].str[:1]
    data['Name_Title'] = data['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
    data['Fam']=data['SibSp']+data['Parch']
    for i in range(0, len(data.index)):
        if data['Fam'][i]>2:
            data['FamGr'][i]=2
        elif data['FamGr'][i]==2:
            data['FamGr'][i]=1
        else:
            data['FamGr'][i]=0
        
    return data

dfProc=procesDat(df)
testProc=procesDat(test)
# edit categorical variables and transform them to numerical
catVar=['Sex', 'Cabin', 'Embarked', 'Name_Title']
for i in catVar:
    dfProc[i].fillna('U', inplace=True)
    testProc[i].fillna('U', inplace=True)
    for j in range(0, len(testProc.index)):
        if testProc['Name_Title'][j]=='Dona.':
            testProc['Name_Title'][j]='Don.'
    dfProc[i]= pLE.fit_transform(dfProc[i])
    testProc[i]= pLE.transform(testProc[i])

# find whith which variables age is corrlated simple model
xVar=['Pclass', 'Sex', 'Cabin','Embarked','NameL', 'FamGr', 'Name_Title']
for i in xVar:
    model=ols(f'Age~C({i})', data=dfProc).fit()
    anova_table = sm.stats.anova_lm(model, typ=1)
    print(anova_table)

# check with multivariate model
mavrMd = ols('Age ~ Pclass + Sex + Cabin + Embarked + NameL + FamGr+Name_Title', dfProc).fit()
mvarTab = sm.stats.anova_lm(mavrMd, typ=2)
print(mvarTab)

# calculate median age based on variables that are likely to be significantly correlated with age
# for manual check of correct median
res = dfProc.groupby(['Pclass', 'FamGr','Name_Title'])['Age'].median().reset_index()
# replace NAs in age with median per group
medPerGr = dfProc.groupby(['Pclass','FamGr','Name_Title'])['Age'].transform('median')
dfProc['Age']=dfProc['Age'].fillna(medPerGr)
# check if NAs left
print(dfProc.isnull().sum())
# median Age for test set
testMedPerGr = testProc.groupby(['Pclass','FamGr','Name_Title'])['Age'].transform('median')
testProc['Age']=testProc['Age'].fillna(testMedPerGr)
print(testProc.isnull().sum())
# for test set fill last NAs with medians from all samples
cols=['Age', 'Fare']
for i in cols:
    testProc[i].fillna(testProc[i].median(), inplace=True)

print(testProc.isnull().sum())
# set variables for random forest
indVar=['Pclass', 'Age', 'Sex','Cabin','Embarked','NameL', 'FamGr', 'Name_Title', 'Fare','Parch', 'SibSp',]
survived=dfProc.Survived
dfFiltr=dfProc[indVar]
testFiltr=testProc[indVar]
# fit random forest train
survived=dfProc.Survived

survMod=RandomForestClassifier(criterion='gini', 
                             n_estimators=1200,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=13,
                             n_jobs=-1)

survMod.fit(dfFiltr, survived)
# show how many were classified correctly
print("%.4f" % survMod.oob_score_)
# show variables importance 
pd.concat((pd.DataFrame(dfFiltr.columns, columns = ['variable']), 
           pd.DataFrame(survMod.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]

# predict test
val_predict=survMod.predict(testFiltr)
# make data with Ids and survival
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': val_predict})
# save
output.to_csv('submission.csv', index=False)

# if need to make sure that output is integer than

#output['Survived']=output['Survived'].astype(int)