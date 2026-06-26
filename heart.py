import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("heart.csv")
print(df.head())


'''
EDA (Exploratory data analysis)'''


print(df.columns)
print(df.shape)
print(df.info())
print(df.describe())



''' EDA AND CLEANING use together'''

print(df.duplicated().sum())

print(df.isnull().sum())



'''
here using matplot for showing graphically instead of sns
'''


print(df['HeartDisease'].value_counts().plot(kind='bar'))
# plt.show()





def plotting(var,num):
    plt.subplot(2,2,num)
    sns.histplot(df[var],kde= True)
plotting('Age',1)
plotting('RestingBP',2)
plotting('Cholesterol',3)
plotting('MaxHR',4)


plt.tight_layout()
# plt.show()



cholesterol = df['Cholesterol'].value_counts()
print(cholesterol)



cholesterol_mean = df.loc[df['Cholesterol'] != 0,'Cholesterol'].mean()

df['Cholesterol'] = df['Cholesterol'].replace(0,cholesterol_mean).round(2)
print(cholesterol_mean)


restingBP_mean = df.loc[df['RestingBP'] != 0, 'RestingBP'].mean()
df['RestingBP'] = df['RestingBP'].replace(0, restingBP_mean).round(2)
print(restingBP_mean)



def plotting(var,num):
    plt.subplot(2,2,num)
    sns.histplot(df[var],kde= True)
plotting('Age',1)
plotting('RestingBP',2)
plotting('Cholesterol',3)
plotting('MaxHR',4)

plt.tight_layout()
# plt.show()


sns.countplot(x= df['Sex'],hue= df['HeartDisease'])
# plt.show()
sns.countplot(x= df['ChestPainType'],hue= df['HeartDisease'])
# plt.show()



sns.boxenplot(x='HeartDisease',y='Cholesterol',data=df)
# plt.show()

sns.heatmap(df.corr(numeric_only=True),annot=True)
# plt.show()







from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score , accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier





# Convert categorical columns into numeric
df = pd.get_dummies(df, drop_first=True)

X = df.drop('HeartDisease',axis=1)
y = df['HeartDisease']





X_train, X_test, y_train, y_test = train_test_split(
 X, y, 
 test_size=0.20, 
 random_state=42)


print("this is X_train\n",X_train.head())
print("this is X_test\n",X_test.head())
print("this is y_train\n",y_train.head())
print("this is y_test\n",y_test.head())




scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)




models = {
    "logistic Regression" : LogisticRegression(),
    "KNN" : KNeighborsClassifier(),
    "Naive bayes" : GaussianNB(),
    "Decision tree" : DecisionTreeClassifier(),
    "SVM" : SVC(),
    "Random forest" : RandomForestClassifier(),
}


result = []


for name,model in models.items():
    model.fit(x_train_scaled,y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    
    
    result.append({
        'model':name,
        'Accuracy': round(accuracy, 4),
        'f1 score':round(f1,4)
        
    })
    
results_df = pd.DataFrame(result)
print(results_df.sort_values(by='Accuracy', ascending=False))

 
 
 
import joblib

joblib.dump(models['KNN'],'KNN_heart.pkl')
joblib.dump(scaler,'scaler.pkl')
joblib.dump(X.columns.tolist(),'columns.pkl')







