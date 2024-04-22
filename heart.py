import pandas as pd
import numpy as np
import pickle

data = pd.read_csv('deploy-ml-model-flask/heart.csv')

from sklearn.model_selection import train_test_split
x = data.drop('HeartDisease',axis=1)
y = data['HeartDisease']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=101)



from sklearn.ensemble import RandomForestClassifier

classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5,
                                       n_estimators=100, oob_score=True)
classifier_rf.fit(x_train, y_train)

try:
    with open('heart.pkl', 'wb') as f:
        pickle.dump(classifier_rf, f)
    print("Pickle file created successfully.")
except Exception as e:
    print("Error occurred while creating pickle file:", e)




"""
X = np.array(df.iloc[:, 0:4])
y = np.array(df.iloc[:, 4:])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y.reshape(-1))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.svm import SVC
sv = SVC(kernel='linear').fit(X_train,y_train)


pickle.dump(sv, open('iri.pkl', 'wb'))
***"""