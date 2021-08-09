import pandas as pd
import numpy as np

df1=pd.read_csv('C:\\Users\\BHASWANTH REDDY\\Desktop\\citizen\\train.csv')

df1=df1.dropna(axis=1)
df1=df1.drop(columns=['appno','application','country.name','docname','doctypebranch','ecli','judgementdate','kpdate','languageisocode','originatingbody_type','parties.0','respondent.0','separateopinion','documentcollectionid=ENG'],axis=1)
df1['itemid']=df1['itemid'].apply(lambda x:int(x[4:7]))
other=df1.groupby('country.alpha2')['country.alpha2'].agg('count').sort_values(ascending=False)
other=other[other<5]

df1['country.alpha2']=df1['country.alpha2'].apply( lambda x:'other' if x in other else x)

delete=[]

df1=df1.drop(columns=delete,axis=1)
#df1=df1.drop(columns=['article=4', 'article=P12',  'article=17', 'article=',  'article=P7', 'article=57', 'article=26', 'article=27', 'article=36', 'article=30', 'article=12', 'article=56', 'article=39', 'article=15', 'article=25', 'article=P6', 'article=19', 'article=32', 'article=33', 'article=53', 'article=28', 'article=43', 'article=52', 'article=P13', 'article=16','applicability=43','applicability=3', 'applicability=22', 'applicability=60', 'applicability=58', 'applicability=25', 'applicability=47', 'applicability=12', 'applicability=38', 'applicability=20', 'applicability=18', 'applicability=24', 'applicability=62', 'applicability=21', 'applicability=23', 'applicability=8', 'applicability=26', 'applicability=53', 'applicability=15', 'applicability=48', 'applicability=14', 'applicability=51', 'applicability=13', 'applicability=5', 'applicability=7', 'applicability=50', 'applicability=52', 'applicability=28', 'applicability=29', 'applicability=31', 'applicability=6', 'applicability=81', 'applicability=66', 'applicability=49', 'applicability=33', 'applicability=63', 'applicability=68', 'applicability=46', 'applicability=19', 'applicability=40', 'applicability=17', 'applicability=32', 'applicability=72', 'applicability=34', 'applicability=35', 'applicability=54', 'applicability=27', 'applicability=16', 'applicability=64', 'applicability=57', 'applicability=56', 'applicability=2', 'applicability=4', 'applicability=67', 'applicability=77', 'applicability=71', 'applicability=59','ccl_article=p6','ccl_article=p7','ccl_article=p12','ccl_article=9','ccl_article=7','ccl_article=46','ccl_article=4','ccl_article=38','ccl_article=34','ccl_article=25','ccl_article=18','ccl_article=17','ccl_article=14','ccl_article=12', 'paragraphs=35-3-b','paragraphs=35-4','paragraphs=57','paragraphs=4-1','paragraphs=P7-4','paragraphs=37-1-c','paragraphs=P4-2-1','paragraphs=9-2','paragraphs=6-3-e','paragraphs=4-2','paragraphs=4-3-d','paragraphs=','paragraphs=17','paragraphs=P4-2-2','paragraphs=P4-2-3','paragraphs=4','paragraphs=5-1-b','paragraphs=5-1-a', 'paragraphs=2-2','paragraphs=P12-1','paragraphs=26', 'paragraphs=27-2', 'paragraphs=27', 'paragraphs=36', 'paragraphs=36-1', 'paragraphs=37-1', 'paragraphs=P7-2', 'paragraphs=35-2', 'paragraphs=5-1-d', 'paragraphs=6-3-a', 'paragraphs=30', 'paragraphs=46-1', 'paragraphs=12', 'paragraphs=P7-1', 'paragraphs=P7-1-1', 'paragraphs=36-2', 'paragraphs=5-2', 'paragraphs=P1-2', 'paragraphs=P1-4', 'paragraphs=56-3', 'paragraphs=56', 'paragraphs=56-1', 'paragraphs=P7-1-2', 'paragraphs=38-1-a', 'paragraphs=37-1-b', 'paragraphs=39', 'paragraphs=15-1', 'paragraphs=15', 'paragraphs=7-2', 'paragraphs=P4-4', 'paragraphs=15-3', 'paragraphs=25-1', 'paragraphs=25', 'paragraphs=37-1-a', 'paragraphs=P6-1', 'paragraphs=P7-5', 'paragraphs=19', 'paragraphs=32', 'paragraphs=P7-3', 'paragraphs=33', 'paragraphs=28-1-a', 'paragraphs=53', 'paragraphs=28', 'paragraphs=35-2-b', 'paragraphs=43', 'paragraphs=4-3-b', 'paragraphs=4-3-a', 'paragraphs=4-3', 'paragraphs=28-3', 'paragraphs=52', 'paragraphs=P13-1', 'paragraphs=P7-4-1', 'paragraphs=16', 'paragraphs=P4-3', 'paragraphs=27-1-b', 'paragraphs=29-1', 'paragraphs=32-2', 'paragraphs=28-1', 'paragraphs=P6-2', 'paragraphs=46-4'],axis=1)
df1=df1.replace(to_replace=-1,value=0.5)


df1=df1.drop(columns=delete,axis=1)
df=df1
dummy1=pd.get_dummies(df['country.alpha2'])
col1=dummy1.drop(columns='other',axis=1).columns.values
dummy2=pd.get_dummies(df['originatingbody_name'])
col2=dummy2.columns.values
df=pd.concat([df,dummy1.drop(columns='other',axis=1),dummy2],axis=1)  
df=df.drop(columns=['country.alpha2','originatingbody_name'],axis=1) 
df.head()

X=np.array(df.drop(columns=['importance'],axis=1))
y=df.importance

from xgboost import XGBClassifier
from mlxtend.classifier import StackingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

cls=[('o',XGBClassifier(max_depth=20,n_estimators=243)),('a',LGBMClassifier(n_estimators=500,num_leaves=42,learning_rate=0.01,class_weight='balanced')),('c',CatBoostClassifier(iterations=500,eta=1))]
cl=[i[1] for i in cls]
mod = StackingClassifier(classifiers=cl,meta_classifier=LGBMClassifier(n_estimators=250,num_leaves=42,class_weight='balanced'))
mod.fit(X,y)
mod.score(X,y)

test=pd.read_csv('C:\\Users\\BHASWANTH REDDY\\Desktop\\citizen\\test.csv')
add=test.appno

test=test.dropna(axis=1)
test=test.drop(columns=['appno','application','country.name','docname','doctypebranch','ecli','judgementdate','kpdate','languageisocode','originatingbody_type','parties.0','respondent.0','separateopinion','documentcollectionid=ENG'],axis=1)
ca=test['country.alpha2']
org=test['originatingbody_name']
test=test.drop(columns=['country.alpha2','originatingbody_name'],axis=1)
test=test.drop(columns=delete,axis=1)
test['itemid']=test['itemid'].apply(lambda x:int(x[4:7]))
df1=df1.replace(to_replace=-1,value=0.5)
#test=test.drop(columns=['article=4', 'article=P12',  'article=17', 'article=',  'article=P7', 'article=57', 'article=26', 'article=27', 'article=36', 'article=30', 'article=12', 'article=56', 'article=39', 'article=15', 'article=25', 'article=P6', 'article=19', 'article=32', 'article=33', 'article=53', 'article=28', 'article=43', 'article=52', 'article=P13', 'article=16','applicability=43','applicability=3', 'applicability=22', 'applicability=60', 'applicability=58', 'applicability=25', 'applicability=47', 'applicability=12', 'applicability=38', 'applicability=20', 'applicability=18', 'applicability=24', 'applicability=62', 'applicability=21', 'applicability=23', 'applicability=8', 'applicability=26', 'applicability=53', 'applicability=15', 'applicability=48', 'applicability=14', 'applicability=51', 'applicability=13', 'applicability=5', 'applicability=7', 'applicability=50', 'applicability=52', 'applicability=28', 'applicability=29', 'applicability=31', 'applicability=6', 'applicability=81', 'applicability=66', 'applicability=49', 'applicability=33', 'applicability=63', 'applicability=68', 'applicability=46', 'applicability=19', 'applicability=40', 'applicability=17', 'applicability=32', 'applicability=72', 'applicability=34', 'applicability=35', 'applicability=54', 'applicability=27', 'applicability=16', 'applicability=64', 'applicability=57', 'applicability=56', 'applicability=2', 'applicability=4', 'applicability=67', 'applicability=77', 'applicability=71', 'applicability=59','ccl_article=p6','ccl_article=p7','ccl_article=p12','ccl_article=9','ccl_article=7','ccl_article=46','ccl_article=4','ccl_article=38','ccl_article=34','ccl_article=25','ccl_article=18','ccl_article=17','ccl_article=14','ccl_article=12', 'paragraphs=35-3-b','paragraphs=35-4','paragraphs=57','paragraphs=4-1','paragraphs=P7-4','paragraphs=37-1-c','paragraphs=P4-2-1','paragraphs=9-2','paragraphs=6-3-e','paragraphs=4-2','paragraphs=4-3-d','paragraphs=','paragraphs=17','paragraphs=P4-2-2','paragraphs=P4-2-3','paragraphs=4','paragraphs=5-1-b','paragraphs=5-1-a', 'paragraphs=2-2','paragraphs=P12-1','paragraphs=26', 'paragraphs=27-2', 'paragraphs=27', 'paragraphs=36', 'paragraphs=36-1', 'paragraphs=37-1', 'paragraphs=P7-2', 'paragraphs=35-2', 'paragraphs=5-1-d', 'paragraphs=6-3-a', 'paragraphs=30', 'paragraphs=46-1', 'paragraphs=12', 'paragraphs=P7-1', 'paragraphs=P7-1-1', 'paragraphs=36-2', 'paragraphs=5-2', 'paragraphs=P1-2', 'paragraphs=P1-4', 'paragraphs=56-3', 'paragraphs=56', 'paragraphs=56-1', 'paragraphs=P7-1-2', 'paragraphs=38-1-a', 'paragraphs=37-1-b', 'paragraphs=39', 'paragraphs=15-1', 'paragraphs=15', 'paragraphs=7-2', 'paragraphs=P4-4', 'paragraphs=15-3', 'paragraphs=25-1', 'paragraphs=25', 'paragraphs=37-1-a', 'paragraphs=P6-1', 'paragraphs=P7-5', 'paragraphs=19', 'paragraphs=32', 'paragraphs=P7-3', 'paragraphs=33', 'paragraphs=28-1-a', 'paragraphs=53', 'paragraphs=28', 'paragraphs=35-2-b', 'paragraphs=43', 'paragraphs=4-3-b', 'paragraphs=4-3-a', 'paragraphs=4-3', 'paragraphs=28-3', 'paragraphs=52', 'paragraphs=P13-1', 'paragraphs=P7-4-1', 'paragraphs=16', 'paragraphs=P4-3', 'paragraphs=27-1-b', 'paragraphs=29-1', 'paragraphs=32-2', 'paragraphs=28-1', 'paragraphs=P6-2', 'paragraphs=46-4'],axis=1)
ml,n=test.shape
test=np.array(test)


def find(arr,n):
    t=np.where(arr==n)[0]
    if(t.size==0):
        return -1
    return t[0]

y_test=[]
for i in range(ml):
    ar1=np.zeros((len(col1),1))
    f1=find(col1,ca[i])
    ar2=np.zeros((len(col2),1))
    f2=find(col2,org[i])        
    if(f1!=-1):
        ar1[f1]=1
    if(f2!=-1):
        ar2[f2]=1
    arr=np.append(test[i],ar1)
    arr=np.append(arr,ar2)
    y_test.append(mod.predict(arr.reshape(1, -1))[0])
y_test=pd.DataFrame(data=y_test,columns=['importance'])

res=pd.concat([add,y_test],axis=1)


print(res)
print('')
print(res['importance'].value_counts())
res.to_csv('C:\\Users\\BHASWANTH REDDY\\Desktop\\citizen\\res.csv',index=False)
