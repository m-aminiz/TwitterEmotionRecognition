# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
from keras.layers import Dense
from keras.utils import to_categorical
from matplotlib.colors import ListedColormap
from matplotlib.colors import ListedColormap
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# Importing the dataset
# dataset = pd.read_csv('Social_Network_Ads.csv')
#
# X = dataset.iloc[:, [2, 3]].values
# y = dataset.iloc[:, 4].values
#
# # Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#
# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# print(y_train)

(trdata , trlabels) , (tesdata , teslabels) = imdb.load_data(num_words=10000)

train_data = trdata[:8000]
train_labels = trlabels[:8000]
test_data = tesdata[:8000]
test_labels = teslabels[:8000]

word_index = imdb.get_word_index()
reverse_word_index = dict ([(value,key) for (key,value) in word_index.items()])
# decoded_review = ' '.join([reverse_word_index.get(i - 3,'?') for i in train_data[20]])
# print(decoded_review)
# print(train_labels[18])

def vectorize_sequence(sequences , dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i , sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

new_train_data = vectorize_sequence(train_data)
new_test_data = vectorize_sequence(test_data)

new_train_label = np.asarray(train_labels).astype('float32')
new_test_label = np.asarray(test_labels).astype('float32')
print('pre process done')


# Fitting classifier to the Training set
# Create your classifier here
print(new_train_label.shape)
classifier = GaussianNB()
classifier.fit(new_train_data,new_train_label)

# Predicting the Test set results
predicted_label = classifier.predict(new_test_data)

showList = []
for j in range(0,5):
    sh = []
    sh.append(' '.join([reverse_word_index.get(i - 3,'?') for i in train_data[100+j]]))
    a = str(new_test_label[100+j])
    sh.append('real label:'+a)
    b = str(predicted_label[100+j])
    sh.append('redicted label:'+b)
    showList.append(np.array(sh))

print(np.array(showList))
print('accuracy :' ,accuracy_score(new_test_label, predicted_label))
# Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(new_test_label, y_pred)
#
# # Visualising the Training set results
#
# X_set, y_set = new_train_data, new_train_label
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('Classifier (Training set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()
#
# # Visualising the Test set results
#
# X_set, y_set = new_test_data, new_test_label
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('Classifier (Test set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()
