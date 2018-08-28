import pandas as pd

patient=pd.read_csv("patient.csv")

print "The first five entries are:"
print patient.head(5)

patient['Gender'] = patient['Gender'].map({'Female': 1, 'Male': 0})
print patient

patient['Has Cancer'] = patient['Has Cancer'].map({True: 1, False: 0})
print patient

patient['Age']=patient['Age'].fillna(patient['Age'].mean()).astype(int)
print patient

patient['Tumor Size']=patient['Tumor Size'].fillna(patient['Tumor Size'].mean()).astype(int)
print patient

patient.to_csv("patient_new.csv", sep='\t', encoding='utf-8')

