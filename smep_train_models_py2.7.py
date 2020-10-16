#source code of smep for training model
#from __future__ import print_function
import numpy as np
import theano
import keras
import pandas as pd

from keras.models import Sequential
from keras.models import load_model
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Flatten, Activation,Embedding
from keras.layers import Conv1D, MaxPooling1D,LSTM
from keras import backend as K
from keras.optimizers import Adam, SGD
from keras.preprocessing import sequence
from keras.constraints import maxnorm
from keras import initializers
from keras.optimizers import RMSprop
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import classification_report

from keras.callbacks import ModelCheckpoint
import shutil, os, sys
from numpy import genfromtxt

theano.config.openmp = True
import datetime as dt
from keras import regularizers

np.set_printoptions(threshold=sys.maxsize)

nb_epoch = 20

batch_size = 128
#epochs = 20

aan=5

in_train_file=sys.argv[1]
in_test_file=sys.argv[2]
in_model_name=sys.argv[3]
max_seq_len=int(sys.argv[4])
class_num  =int(sys.argv[5])

prefix="Train_" + str(in_train_file) + "_Test_" + str(in_test_file) +"_flag_"+ str(in_model_name)

print('Loading data...')

mulu_logs="LOGs/"
mulu_models="MODELs/"

#NUM_THREADS=4

if not os.path.exists(mulu_models):
    os.makedirs(mulu_models)

if not os.path.exists(mulu_logs):
    os.makedirs(mulu_logs)

tar_log_file=mulu_logs + "LOG_" + prefix
model_filepath= mulu_models + "Best_model_" + prefix + ".h5"

infile_train=in_train_file
infile_test=in_test_file

print infile_train
print infile_test

times1 = dt.datetime.now()

train_data = pd.read_csv(infile_train, index_col = False, header=None)
test_data = pd.read_csv(infile_test, index_col = False, header=None)

print train_data.shape
print test_data.shape

y_test=test_data[0]
x_test_ori=test_data[1]

x_test=[]
for pi in x_test_ori:
	nr=pi.split(' ')[0:-1]
	#print nr
	ndata=map(int,nr)
	x_test.append(ndata)

x_test=np.array(x_test)

y_train=train_data[0]
x_train_ori=train_data[1]

x_train=[]
for pi in x_train_ori:
	nr=pi.split(' ')[0:-1]
	ndata=map(int,nr)
	x_train.append(ndata)

x_train=np.array(x_train)

times2 = dt.datetime.now()

print('Time spent: '+ str(times2-times1))

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, class_num)
y_test = np_utils.to_categorical(y_test, class_num)

x_train=sequence.pad_sequences(x_train,maxlen=max_seq_len)
x_test=sequence.pad_sequences(x_test,maxlen=max_seq_len)

model = Sequential()
model.add(Embedding(aan, aan, input_length=max_seq_len))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu',padding='same',kernel_initializer='random_uniform', kernel_constraint=maxnorm(3),bias_constraint=maxnorm(3)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.20))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu',padding='same',kernel_initializer='random_uniform', kernel_constraint=maxnorm(3),bias_constraint=maxnorm(3)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.20))

model.add(Flatten())

model.add(Dense(64, activation='relu',kernel_initializer='random_uniform', kernel_constraint=maxnorm(3),bias_constraint=maxnorm(3)))
model.add(Dropout(0.20))
model.add(Dense(32, activation='relu',kernel_initializer='random_uniform', kernel_constraint=maxnorm(3),bias_constraint=maxnorm(3)))
model.add(Dropout(0.20))
model.add(Dense(class_num, activation='softmax'))

checkpoint = ModelCheckpoint(model_filepath, monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')
callbacks_list = [checkpoint]

fout=open(tar_log_file, 'w', 0)

print('Training')
for i in range(nb_epoch):
	print('Epoch', i, '/', nb_epoch)
	OPTIMIZER=Adam()
	model.compile(loss='categorical_crossentropy',optimizer=OPTIMIZER,metrics=['accuracy'])
	Result=model.fit(x_train, y_train, epochs=1, callbacks=callbacks_list, batch_size=batch_size, validation_data=(x_test, y_test), class_weight='auto', verbose = 1)
	loss_and_metrics = model.evaluate(x_test, y_test)
	print >>fout, "Epoch ",i,"  ",prefix," loss and metrics Test : ",loss_and_metrics
	model.reset_states()

del model

model = load_model(model_filepath)
loss_and_metrics = model.evaluate(x_test, y_test)
print >>fout, "\n\nFinal loss and metrics : ",loss_and_metrics,"\n\n"

y_pred = model.predict(x_test)

auc=roc_auc_score(y_test.flatten(),y_pred.flatten())
print >>fout, "auc"
print >>fout, auc

y_pred = np.argmax(y_pred, axis=1) 
y_real = np.argmax(y_test, axis=1)
	
print >>fout, "classification_report"
print >>fout, classification_report(y_real, y_pred)

fout.close()
sys.exit()
