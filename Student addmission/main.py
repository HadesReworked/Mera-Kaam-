import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv('ST_Admission.csv')

print(df.isnull().sum())

age = df['Age'].fillna(df['Age'].mean())
print(age.isnull().sum())


gender=df['Gender'].dropna()
print(gender.isnull().sum())

test=df['Admission Test Score'].dropna()
print(test.isnull().sum())

percent=df['High School Percentage'].dropna()
print(percent.isnull().sum())

city=df['City'].dropna()
print(city.isnull().sum())

status=df['Admission Status'].dropna()
print(status.isnull().sum())

dropped=df.dropna()
print(dropped.info())
print(dropped.head())
print(dropped.tail())
print(dropped.describe())


plt.figure(figsize=(12, 9))
plt.title('Students Admission Record')
plt.xlabel('Marks')
plt.ylabel('Percentage')
plt.grid(True)
plt.show()



