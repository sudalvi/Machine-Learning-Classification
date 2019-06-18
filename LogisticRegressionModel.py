import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#Read CSV file
data = pd.read_csv('Social_Network_Ads.csv')
X = data.iloc[:,[2,3]].values
Y = data.iloc[:,4].values

#Data Preprocessing Step
st_x_train = StandardScaler()
X= st_x_train.fit_transform(X)

#Creating Model By Fitting Data in LogisticRegression
classifier = LogisticRegression()
classifier.fit(X, Y)

#Creating Pickel File
pickle_out = open("LogisticRegressionPickelFile","wb")
pickle.dump(classifier, pickle_out)
