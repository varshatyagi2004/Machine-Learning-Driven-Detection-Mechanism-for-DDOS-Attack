# DDOS attack Detection
### Creating a platform that can detect DDOS attacks

# installing modules
! pip install pandas numpy matplotlib seaborn scikit-learn


                          #importing modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,ConfusionMatrixDisplay
from sklearn import metrics

# reading Data

df=pd.read_csv("/content/dataset_sdn.csv")
df

## I. Data Preprocessing
# Information about data
df.info()

df.describe()

# column names
column_names= df.columns
column_names

# Null values sum
df.isnull().sum().plot.bar()
plt.title("NULL Values for each column ")
plt.xlabel("Column names")
plt.ylabel("Count")


# Dropping rows having null values
df=df.dropna()


df.info()

# Getting unique destination
uniq_dest=df['dst'].unique()
total_dst=len(uniq_dest)
print("Total destination : ", total_dst)
print("Different destination : ",uniq_dest)

# Doing analysis for malicious and normal traffic
gp=df.groupby('label')['label'].count()
plt.bar(list(gp.index),list(gp.values),color=['g','r'])
plt.xticks(list(gp.index))
plt.xlabel("Traffic label")
plt.ylabel("Count")
plt.title("Traffic for normal and Malicious traffic")

ip_addr=df[df['label']==0].groupby('dst').count()['label'].index
normal_traffic=df.groupby(['dst','label']).size().unstack().fillna(0)[0]
attack_traffic=df.groupby(['dst','label']).size().unstack().fillna(0)[1]
plt.barh(ip_addr,normal_traffic,color='g', label='Normal Traffic')
plt.barh(ip_addr,attack_traffic,color='r', label='Attack Traffic')
plt.legend()
plt.xlabel("Count")
plt.ylabel("Destination IP Adresses")
plt.title("Attack and Normal traffic ")

# Columns containing object(string) type data
# Port no column also does not do much so ignoring it also
object_col= list(df.select_dtypes(include=['object']).columns)
object_col=object_col+['port_no']
print(object_col)
data=df.drop(columns=object_col)

# seperating data based on protocol
udp_df = df[df['Protocol']=='UDP'].drop(columns=object_col)
tcp_df = df[df['Protocol']=='TCP'].drop(columns=object_col)
icmp_df = df[df['Protocol']=='ICMP'].drop(columns=object_col)
udp_df

def show_confusion_matrix(model,data,label,pred_label):
  class_names=['Normal','Attack']

  cm = confusion_matrix(label, pred_label)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
  disp.plot(cmap=plt.cm.Blues, values_format='.2f')  # Plot confusion matrix
  plt.title('Confusion Matrix')
  plt.xlabel('Actual')
  plt.ylabel('Predicted')
  plt.show()  # Show plot

  print('\n')
  print(classification_report(label, pred_label))
  print(confusion_matrix(label,pred_label))
  print('\n')

## II. Data Model building
# importin modules for train test split
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

### GMM
#### UDP Protocol
# splitting traing and testing data
udp_train,udp_test, udp_train_label, udp_test_label= train_test_split(udp_df[udp_df.columns[:-1]],udp_df['label'],test_size=0.3,random_state=42)

gmm= GaussianMixture(n_components=2)
gmm.fit(udp_train)
print("GMM Accuracy training data : ",metrics.accuracy_score(udp_train_label, gmm.predict(udp_train)))
print("GMM Accuracy testing data : ", metrics.accuracy_score(udp_test_label,gmm.predict(udp_test)))

#### TCP Protocol
# splitting traing and testing data
tcp_train,tcp_test, tcp_train_label, tcp_test_label= train_test_split(tcp_df[tcp_df.columns[:-1]],tcp_df['label'],test_size=0.3,random_state=42)

gmm= GaussianMixture(n_components=2)
gmm.fit(tcp_train)
print("GMM Accuracy training data : ",metrics.accuracy_score(tcp_train_label, gmm.predict(tcp_train)))
print("GMM Accuracy testing data : ", metrics.accuracy_score(tcp_test_label,gmm.predict(tcp_test)))
print("GMM Precision training data : ", metrics.precision_score(tcp_train_label, gmm.predict(tcp_train)))
print("GMM Precision testing data : ", metrics.precision_score(tcp_test_label,gmm.predict(tcp_test)))

actual = np.random.binomial(1, 0.9, size = 1000)
predicted = np.random.binomial(1, 0.9, size = 1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()

#### ICMP Protocol
# splitting traing and testing data
icmp_train,icmp_test, icmp_train_label, icmp_test_label= train_test_split(icmp_df[icmp_df.columns[:-1]],icmp_df['label'],test_size=0.3,random_state=42)

# GMM model

gmm= GaussianMixture(n_components=2)
gmm.fit(icmp_train)
print("GMM Accuracy training data : ",metrics.accuracy_score(icmp_train_label, gmm.predict(icmp_train)))
print("GMM Accuracy testing data : ", metrics.accuracy_score(icmp_test_label,gmm.predict(icmp_test)))


##### As we can see accuracy is not good for GMM model
## Linear Perceptron
from sklearn.linear_model import Perceptron


#### UDP
model = Perceptron(random_state=67)
model.fit(udp_train,udp_train_label)
model.score(udp_train,udp_train_label)

#### TCP
model = Perceptron(random_state=1)
model.fit(tcp_train,tcp_train_label)
model.score(tcp_train,tcp_train_label)

#### ICMP

model = Perceptron(random_state=1)
model.fit(icmp_train,icmp_train_label)
model.score(icmp_train,icmp_train_label)

## Multi-Layer perceptron
from sklearn.neural_network import MLPClassifier

#### UDP
clf= MLPClassifier(hidden_layer_sizes=(16,10),
                  random_state=5,
                  learning_rate_init=0.01)
clf.fit(udp_train,udp_train_label)
metrics.accuracy_score(clf.predict(udp_test), udp_test_label)

#### TCP
clf= MLPClassifier(hidden_layer_sizes=(18,12),
          random_state=5,
          learning_rate_init=0.01)
clf.fit(tcp_train,tcp_train_label)
print(metrics.accuracy_score(clf.predict(tcp_test), tcp_test_label))


#### ICMP
clf= MLPClassifier(hidden_layer_sizes=(16,10),
                  random_state=5,
                  learning_rate_init=0.01)
clf.fit(icmp_train,icmp_train_label)
metrics.accuracy_score(clf.predict(icmp_test), icmp_test_label)

## KNN
# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

#### UDP
# Create feature and target arrays
X = udp_train
y = udp_train_label


knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X, y)

# Calculate the accuracy of the model
print(knn.score(udp_test, udp_test_label))

#### TCP
X = tcp_train
y = tcp_train_label


knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X, y)

# Calculate the accuracy of the model
print(knn.score(tcp_test, tcp_test_label))

#### ICMP
X = icmp_train
y = icmp_train_label


knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X, y)

# Calculate the accuracy of the model
print(knn.score(icmp_test, icmp_test_label))

## SVM
from sklearn import svm

#### UDP
clf= svm.SVC(kernel='poly')
clf.fit(udp_train,udp_train_label)
clf.score(udp_test,udp_test_label)

#### TCP
clf= svm.SVC()
clf.fit(tcp_train,tcp_train_label)
metrics.accuracy_score(clf.predict(tcp_test),tcp_test_label)

#### ICMP
clf= svm.SVC()
clf.fit(icmp_train,icmp_train_label)
metrics.accuracy_score(clf.predict(icmp_test),icmp_test_label)

## Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

#### UDP
# create regressor object
udp_rf = RandomForestRegressor()

# fit the regressor with x and y data
udp_rf.fit(udp_train,udp_train_label)
predi = udp_rf.predict(udp_test)
accuracy_score(predi.round(), udp_test_label)


#
plt.bar(udp_df.columns[:-1], udp_rf.feature_importances_)
plt.xticks(rotation=45)
plt.title("Feature Importance in deciding class Label")
plt.xlabel("Feautre names")
plt.ylabel("Importance ")

#### TCP
# create regressor object
tcp_rf = RandomForestRegressor()

# fit the regressor with x and y data
tcp_rf.fit(tcp_train,tcp_train_label)
predi = tcp_rf.predict(tcp_test)
accuracy_score(predi.round(), tcp_test_label)


plt.bar(tcp_df.columns[:-1], tcp_rf.feature_importances_)
plt.xticks(rotation=45)
plt.title("Feature Importance in deciding class Label")
plt.xlabel("Feautre names")
plt.ylabel("Importance ")


#### ICMP
# create regressor object
icmp_rf = RandomForestRegressor()

# fit the regressor with x and y data
icmp_rf.fit(icmp_train,icmp_train_label)
accuracy_score(icmp_rf.predict(icmp_test).round(), icmp_test_label)

plt.bar(icmp_df.columns[:-1], icmp_rf.feature_importances_)
plt.xticks(rotation=45)
plt.title("Feature Importance in deciding class Label")
plt.xlabel("Feautre names")
plt.ylabel("Importance ")


##### As Accuracy for Random Forest is high . So choosing this model