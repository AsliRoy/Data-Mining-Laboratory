import pandas as pd
import numpy
df=pd.read_csv("table.csv")
print df
df['F1+'] = df['F11'] + df['F10']
df['F0+'] = df['F01'] + df['F00']
df['F+1'] = df['F11'] + df['F01']
df['F+0'] = df['F10'] + df['F00']
df['N'] = df["F1+"] + df["F0+"]

#Symmetric measures
df['correlation']=(df['N']*df['F11']-df['F1+']*df['F+1'])/numpy.sqrt((df['F1+'].astype(float)*df['F+1'].astype(float)*df['F0+'].astype(float)*df['F+0'].astype(float)))
df['odds ratio']=df['F11']*df['F00']/(df['F10']*df['F01'])
df['kappa']=(df['N']*df['F11']+df['N']*df['F00']-df['F1+']*df['F+1']-df['F0+']*df['F+0'])/(df['N']*df['N']-df['F+1']*df['F1+']-df['F0+']*df['F+0'])
df['interest']=df['N']*df['F11']/(df['F1+']*df['F+1'])
df['piatetsky shapiro']=df['F11']/df['N']-df['F1+']*df['F+1']/(df['N']*df['N'])
df['streength']=((df['F11']+df['F00'])/(df['F1+']*df['F+1']+df['F0+']*df['F+0']))*((df['N']-df['F1+']*df['F+1']-df['F0+']*df['F+0'])/(df['N']-df['F11']-df['F00']))
df['jaccard']=df['F11']/(df['F1+']*df['F+1']-df['F11'])

#Assymetic measures
df['Laplace'] = (df["F11"]+1)/(df['F1+']+2)
df['Conviction'] = (df["F1+"]*df["F+0"])/(df['N']*df['F10'])
df['Added value'] = (df['F11']/df['F1+'])-(df['F+1']/df['N'])
df['Certainty factor'] = ((df['F11']/df['F1+'])-(df['F+1']/df['N']))/(1-(df['F1+']/df['N']))
df['Gini index'] = (df['F1+']/df['N'])*((df['F11']/df['F1+'])*(df['F11']/df['F1+'])+(df['F10']/df['F1+'])*(df['F10']/df['F1+']))-((df['F+1']/df['N'])*(df['F+1']/df['N']))+(df['F0+']/df['N'])*((df['F01']/df['F0+'])*(df['F01']/df['F0+'])+(df['F00']/df['F0+'])*(df['F00']/df['F1+']))-((df['F+0']/df['N'])*(df['F+0']/df['N']))
df['J Measure'] = (df['F11']/df['N'])*numpy.log2((df['N']*df['F11'])/(df['F+1']*df['F1+']))+(df['F10']/df['N'])*numpy.log2((df['N']*df['F10'])/(df['F1+']*df['F+0']))
df['Mutual Information'] = ((df['F11']/df['N'])*numpy.log2(df['F11']/(df['F1+']*df['F+1']))+(df['F10']/df['N'])*numpy.log2(df['F10']/(df['F1+']*df['F+0']))+(df['F01']/df['N'])*numpy.log2(df['F01']/(df['F0+']*df['F+1']))+(df['F00']/df['N'])*numpy.log2(df['F00']/(df['F0+']*df['F+0'])))-((df['F1+']/df['N']*numpy.log2(df['F1+']/df['N']))+(df['F0+']/df['N']*numpy.log2(df['F0+']/df['N'])))



#Writing the dataframe values to a CSV file
df.to_csv("measures.csv", sep='\t', encoding='utf-8')
print df



