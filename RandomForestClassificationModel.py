import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Reading Data From CSV file
data  = pd.read_csv("Social_Network_Ads.csv")
X = data.iloc[:,[2,3]].values
Y = data.iloc[:,4].values

sc = StandardScaler()
X_train = sc.fit_transform(X)
X_test = sc.fit_transform(X)

# Creating Classifier Object
classifier  = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, Y)

# Creating Pickle File
pickle_out = open("RandomForestClassificationPickleFile", "wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()
