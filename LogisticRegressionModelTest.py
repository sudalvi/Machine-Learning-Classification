import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
data = pd.read_csv('Social_Network_Ads.csv')
X = data.iloc[:,[2,3]].values
Y = data.iloc[:,4].values

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
st_x_train = StandardScaler()
st_x_test = StandardScaler()
X_train = st_x_train.fit_transform(X_train)
X_test = st_x_test.fit_transform(X_test)

#Loading Pickle File
pickle_in = open("LogisticRegressionPickelFile", "rb")
classifier = pickle.load(pickle_in)
y_pred = classifier.predict(X_test)

#Ploting Graph
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()-1,step=0.01),
                     np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()-1,step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set== j,0], X_set[Y_set == j,1], c = ListedColormap(('red','green'))(i),label=j)

plt.title('Logistic Regression')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()