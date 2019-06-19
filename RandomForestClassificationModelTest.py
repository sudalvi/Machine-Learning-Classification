import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# Reading Data from CSV File
data  = pd.read_csv("Social_Network_Ads.csv")
X = data.iloc[:,[2,3]].values
Y = data.iloc[:,4].values

sc = StandardScaler()
X_train = sc.fit_transform(X)
X_test = sc.fit_transform(X)

# Loading Pickle
pickle_in = open("RandomForestClassificationPickleFile","rb")
classifier = pickle.load(pickle_in)

# Ploting Graph
X_set, Y_set = X_train, Y
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()-1, step=0.01),
                     np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()-1, step=0.01))
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