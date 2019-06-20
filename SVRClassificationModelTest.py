import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from matplotlib.colors import ListedColormap

# Reading Data from CSV file
salary_data = pd.read_csv("Social_Network_Ads.csv")
X = salary_data.iloc[:,[2,3]].values
Y = salary_data.iloc[:,4].values

# Spliting Data into Train and test Part
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=0)

# Preprocessing Data
sc =  StandardScaler()
X_train = sc.fit_transform(X_train)
X_test =  sc.transform(X_test)

# Open Pickle File
pickleRead = open("SVRClassificationPickleFile", "rb")
classifier = pickle.load(pickleRead)

# PLoting Graph
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()-1, step=0.1),
                     np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()-1, step=0.1))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set== j,0], X_set[Y_set == j,1], c = ListedColormap(('red','green'))(i),label=j)

plt.title('Random Forest Classifier')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()

