import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
pLE=preprocessing.LabelEncoder()


df=pd.read_csv('../input/titanic/train.csv')
test=pd.read_csv("../input/titanic/test.csv")

df['Name_Title'] = df['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0]) # make column with titles

cols=['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']
def procesTrain(df):
    data=df[cols]
    for i in cols:
        if is_numeric_dtype(data[i])==True:
            data[i].fillna(data[i].median(), inplace=True)
        else:
            data[i].fillna('U', inplace=True)
    return data

dfFiltr=procesTrain(df)
print(dfFiltr.isnull().sum())

dfFiltr=procesTrain(df)
print(dfFiltr.isnull().sum()) # make sure that no NAs left

def procesTest(df):
    data=df[cols]
    for i in cols:
        if is_numeric_dtype(data[i])==True:
            data[i].fillna(data[i].median(), inplace=True)
        else:
            data[i].fillna('U', inplace=True)
    return data

dfFiltrTest=procesTest(test)

print(dfFiltrTest.isnull().sum())

catVar=['Sex', 'Embarked']
for i in catVar:
    dfFiltr[i]= pLE.fit_transform(dfFiltr[i])
    dfFiltrTest[i]= pLE.transform(dfFiltrTest[i])
    print(pLE.classes_)

survived=df.Survived

survMod=RandomForestClassifier(criterion='gini', 
                             n_estimators=900,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=13,
                             n_jobs=-1)

survMod.fit(dfFiltr, survived)

print("%.4f" % survMod.oob_score_) # print percentage

# see what variables are important
pd.concat((pd.DataFrame(dfFiltr.columns, columns = ['variable']), 
           pd.DataFrame(survMod.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]

# predict and save
val_predict=survMod.predict(dfFiltrTest)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': val_predict})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

