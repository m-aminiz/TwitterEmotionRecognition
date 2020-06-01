from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
from keras.layers import Dense
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

(train_data , train_labels) , (test_data , test_labels) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()
reverse_word_index = dict ([(value,key) for (key,value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3,'?') for i in train_data[18]])
# print(decoded_review)
# print(train_labels[18])

def vectorize_sequence(sequences , dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i , sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results
print(train_data)
new_train_data = vectorize_sequence(train_data)
new_test_data = vectorize_sequence(test_data)
print(new_train_data)
new_train_label = np.asarray(train_labels).astype('float32')
new_test_label = np.asarray(test_labels).astype('float32')
print(new_train_label)

model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))


model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
# model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss=losses.binary_crossentropy,metrics=[metrics.binary_accuracy])

final_train_data = new_train_data[:10000]
evaluation_data = new_train_data[10000:]
final_train_label = new_train_label[:10000]
evaluation_label = new_train_label[10000:]

h = model.fit(evaluation_data,evaluation_label,epochs=20,batch_size=512,validation_data=(final_train_data,final_train_label))

history_dict = h.history
loss_value = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1,len(loss_value)+1)

plt.plot(epochs,loss_value,'bo',label='training loss')
plt.plot(epochs,loss_value,'b',label='validation loss')
plt.title('training and validation loss')
plt.xlabel('epochs')
plt.xlabel('loss')
plt.legend()
plt.show()

# plt.clf()

acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs,acc_values,'bo',label='training acc')
plt.plot(epochs,acc_values,'bo',label='validation acc')
plt.title('training and validation accuracy')
plt.xlabel('epochs')
plt.xlabel('loss')
plt.legend()
plt.show()


model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(new_train_data , new_train_label , epochs=4 , batch_size=512)
results = model.evaluate(new_test_data,new_test_label)

predicted_label = model.predict(new_test_data)
print(predicted_label)
# print(predicted_label[18])
# print(len(predicted_label))
# print(len(predicted_label[18]))


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
print('accuracy :' ,results[1])
