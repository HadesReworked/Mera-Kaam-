import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Medicaldataset.csv")

print(df.head())
print(df.tail())
print(print(df.duplicated().sum()))
print(df.info())
print( df.isnull().sum())
print( df.describe(include='all'))

df_cleaned = df[
    (df["Heart rate"] >= 30) & (df["Heart rate"] <= 220) &
    (df["Systolic blood pressure"] >= 70) & (df["Systolic blood pressure"] <= 200) &
    (df["Diastolic blood pressure"] >= 40) & (df["Diastolic blood pressure"] <= 130) &
    (df["CK-MB"] <= 100) &
    (df["Troponin"] <= 5)
].copy()

df_cleaned["Result_binary"] = df_cleaned["Result"].map({"negative": 0, "positive": 1})
corr=df_cleaned.corr(numeric_only=True)
print (df_cleaned.groupby("Result").mean())
print( corr)

age_bins = [0, 30, 40, 50, 60, 70, 80, 100]
age_labels = ['<30', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
df_cleaned['AgeGroup'] = pd.cut(df_cleaned['Age'], bins=age_bins, labels=age_labels)

heart_attacks = df_cleaned[df_cleaned['Result'] == 'positive']

age_group_counts = heart_attacks['AgeGroup'].value_counts().sort_index()

age_group_counts.plot(kind='bar', title='Heart Attack Cases by Age Group', ylabel='Cases', xlabel='AgeGroup', color='skyblue')

cols_to_plot = ["Age", "Heart rate", "Systolic blood pressure", "Diastolic blood pressure", "Blood sugar", "CK-MB", "Troponin"]
for col in cols_to_plot:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df_cleaned, x=col, hue="Result", kde=True, bins=30, palette="Set1")
    plt.title(f"Distribution of {col} by Result")
    plt.tight_layout()
    plt.show()

heart_attack_by_age = df_cleaned[df_cleaned['Result_binary'] == 1].groupby('AgeGroup', observed=True).size()
plt.figure(figsize=(8, 5))
sns.barplot(x=heart_attack_by_age.index, y=heart_attack_by_age.values,hue = heart_attack_by_age.index, palette='Reds',legend=False)
plt.title('Number of Heart Attacks by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Number of Positive Cases')
plt.show()

systolic_by_age = df_cleaned.groupby('AgeGroup', observed=True)['Systolic blood pressure'].mean().reset_index()

plt.figure(figsize=(8, 5))
plt.plot(systolic_by_age['AgeGroup'], systolic_by_age['Systolic blood pressure'], 
         marker='o', linestyle='-', color='darkred', label='Avg Systolic BP')

plt.title('Average Systolic Blood Pressure by AgeGroup')
plt.xlabel('AgeGroup')
plt.ylabel('Systolic Blood Pressure (mmHg)')
plt.grid(True)
plt.show()

diastolic_by_age = df_cleaned.groupby('AgeGroup', observed=True)['Diastolic blood pressure'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=diastolic_by_age, x='AgeGroup', y='Diastolic blood pressure',hue = 'AgeGroup', palette='viridis',legend=False)
plt.title('Average Diastolic Blood Pressure by AgeGroup', fontsize=14)
plt.xlabel('AgeGroup', fontsize=12)
plt.ylabel('Average Diastolic Blood Pressure (mmHg)', fontsize=12)
plt.show()


blood_sugar_avg = df_cleaned.groupby('AgeGroup', observed=True)['Blood sugar'].mean().reset_index()

plt.figure(figsize=(8, 5))
plt.plot(blood_sugar_avg['AgeGroup'], blood_sugar_avg['Blood sugar'], 
         marker='o', linestyle='-', color='darkred', label='Avg Systolic BP')

plt.title('Average Blood Sugar by Age Group')
plt.xlabel('AgeGroup')
plt.ylabel('Average Blood Sugar (mg/dL)')
plt.grid(True)
plt.show()

gender_mapping = {1: 'Males', 0: 'Females'}
df_cleaned['Gender'] = df_cleaned['Gender'].map(gender_mapping)

heart_attacks_by_gender = df_cleaned[df_cleaned['Result_binary'] == 1].groupby('Gender').size()

plt.figure(figsize=(6, 4))
sns.barplot(x=heart_attacks_by_gender.index, y=heart_attacks_by_gender.values,hue= heart_attacks_by_gender.index, palette='Blues',legend=False)

plt.title('Number of Heart Attacks by Gender')
plt.ylabel('Number of Heart Attacks')
plt.xlabel('Gender')
plt.tight_layout()
plt.show()