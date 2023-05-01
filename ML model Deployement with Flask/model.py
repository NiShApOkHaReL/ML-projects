import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('IRIS.csv')

print(df.head())

x = df[['sepal_length','sepal_width','petal_length','petal_width']]
y = df['species']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=50)

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = RandomForestClassifier()

classifier.fit(x_train, y_train)

pickle.dump(classifier, open("model.pkl", "wb"))
