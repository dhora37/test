# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 22:57:35 2019

@author: HOME
"""
#ARM

#!pip install mlxtend

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

dataset=[['Milk','Onion','Nutmeg','Kidney Beans','Eggs','Yogurt'],
         ['Dill','Onion','Nutmeg','Kidney Beans','Eggs','Yougurt'],
         ['Milk','Apple','Kidney Beans','Eggs'],
         ['Milk','Unicorn','Corn','Kidney Beans','Yougurt'],
         ['Corn','Onion','Onion','Kidney Beans','Ice cream','Eggs']]

te=TransactionEncoder()

te_ary=te.fit(dataset).transform(dataset)
te_ary

df=pd.DataFrame(te_ary,columns=te.columns_)
df

frequent_itemsets=apriori(df,min_support=0.6,use_colnames=True)
frequent_itemsets

from mlxtend.frequent_patterns import association_rules

a_r=association_rules(frequent_itemsets,metric="confidence",min_threshold=0.7)

rules=association_rules(frequent_itemsets,metric="lift",min_threshold=1.2)
rules

rules=rules[['antecedents','consequents','support','confidence','lift']]
rules

rules["antecedent_len"]=rules["antecedents"].apply(lambda x:len(x))
rules

rules[(rules['antecedent_len']>=2)&
      (rules['confidence']>0.75)&
      (rules['lift']>1.2)]
      
eggs_kbs=rules[rules['antecedents']=={'Eggs','Kidney Beans'}]
eggs_kbs

res=rules
res=res[['antecedents','consequents','support']]
res

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df=pd.read_csv('groceries_200.CSV')

df.head()

df.isnull().sum()

df.fillna(0,inplace=True)
df.head()

arr=np.array(df.values)
all_items=np.unique(arr[~(arr==0)])
print(all_items)

basket=pd.DataFrame(0,index=np.arange(len(df)),columns=all_items)
basket.head()

for col in df.columns: #columnwise
    t_f=(df[col].values!=0)#
    itms=df.iloc[t_f,col]#all the elements which are not zero in col
    print(itms)
    grocareies_10
    #|| first element of every basket
    indx=itms.index #save the index of above elements
    for i in range(len(itms)):
        basket.loc[indx[i]][itms.iloc[i]]+=1
        
basket.head()
        
frequent_itemsets=apriori(basket.min_support=0.02,use_colnames=True)
frequent_itemsets.head()
    
frequent_itemsets.tail()

#frequent_itemsets=apriori(basket.min_support=0.02,use_colnames=True)
frequent_itemsets.head()

frequent_itemsets['length']=frequent_itemsets['itemsets'].apply(lambda x:len(x))

frequent_itemsets.tail()

rules=association_rules(frequent_itemsets.metric="lift",min_threshold=1)

rules.head()

len(rules)
 
rules=association_rules(frequent_itemsets.metric="lift",min_threshold=5)

len(rules)

rules=association_rules(frequent_itemsets.metric="lift",min_threshold=1)
rules[rules['cofidance']>0.7]

rules=association_rules(frequent_itemsets.metric="confidence",min_threshold=0.5)
rules.head()

rules=association_rules(frequent_itemsets,min_threshold=0.5)
rules.head()

rules=association_rules(frequent_itemsets,min_threshold=2.0)
rules.head()

rules=association_rules(frequent_itemsets.metric="support",min_threshold=0.01)
rules.head()

import pandas as pd

dict={'itemsets':[['177','176'],['177','179']
['176','178'],['176',179],
['93','100'],['177',178'],
['177','176','178']],















































































































































































































































