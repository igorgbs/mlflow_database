from sklearn import datasets, tree, neighbors
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import warnings
import sys

#arguments to the classifier

classifier_selection = int(sys.argv[1]) if len(sys.argv) > 1 else 1


iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.5)

#classifiers
if classifier_selection == 1:
	classifier = tree.DecisionTreeClassifier()
if classifier_selection == 2:	
	classifier = neighbors.KNeighborsClassifier()


#fit function
classifier.fit(x_train,y_train)


#predictions
predictions = classifier.predict(x_test)

#accuracy
accuracy = accuracy_score(y_test,predictions)


#mlflow
#mlflow.set_tracking_uri('sqlite:///C:\\Users\\igorg_000\\Desktop\\Petrobras_Intelipetro\\mlflow-master\\examples\\example_1_igor\\teste2.db')
mlflow.set_tracking_uri('mysql+pymysql://root:@localhost:3306/mlflow')

mlflow.start_run()
mlflow.log_metric("accuracy", accuracy)


print("Accuracy Score:", accuracy)
