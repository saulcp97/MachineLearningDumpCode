#Other Trainings (FAST)
import sklearn
from sklearn import tree
import numpy as np
from sklearn.cross_validation import train_test_split

data = np.loadtxt('dataTrain.csv', delimiter=',', skiprows=0, usecols=range(1,44))
#The first column of data is the label
labels = data[:,0]
features = data[:,1:]



clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
tree.plot_tree(clf)