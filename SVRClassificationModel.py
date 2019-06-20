import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle

salary_data = pd.read_csv("Social_Network_Ads.csv")
X = salary_data.iloc[:,[2,3]].values
Y = salary_data.iloc[:,4].values

# Data preprocessing
sc = StandardScaler()
X = sc.fit_transform(X)

# Creating Classifier
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X, Y)

# Creating Pickle File
pickleFile = open("SVRClassificationPickleFile","wb")
pickle.dump(classifier, pickleFile)
pickleFile.close()

