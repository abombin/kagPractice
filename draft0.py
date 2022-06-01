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
            data['FamGr'][i]=1
        
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

# find whith which variables age is corrlated
xVar=['Pclass', 'Cabin','Embarked','NameL', 'FamGr', 'Name_Title']
for i in xVar:
    model=ols(f'Age~C({i})', data=dfProc).fit()
    anova_table = sm.stats.anova_lm(model, typ=1)
    print(anova_table)