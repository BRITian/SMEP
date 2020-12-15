#source code of smep for prediction

import os
import numpy as np
import theano
import keras
import pandas as pd
from keras.utils import np_utils
from keras.models import load_model
import shutil, os, sys
from numpy import genfromtxt
from keras.preprocessing import sequence
theano.config.openmp = True
import datetime as dt
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=4, suppress=True)
#epochs = 20

infile_prediction=sys.argv[1]
in_model=sys.argv[2]
max_seq_len=int(sys.argv[3])

tar_dir_pres="pres/"
if not os.path.exists(tar_dir_pres):
    os.makedirs(tar_dir_pres)

train_data = pd.read_csv(infile_prediction, index_col = False, header=None)
x_train_ori=train_data[1]
#y_train_ori=train_data[0]

x_train=[]
for pi in x_train_ori:
	nr=pi.split(' ')[0:-1]
	ndata=map(int,nr)
	x_train.append(ndata)
x_train=np.array(x_train)
x_train=sequence.pad_sequences(x_train,maxlen=max_seq_len)

#print x_train

tar_log_file="./pres/"+"Pres."+infile_prediction +"_model_"+in_model

fout=open(tar_log_file, 'w', 0)
model_filepath= in_model 

#print infile_train

model = load_model(model_filepath)

y_pred = model.predict(x_train)	
tar_pre = np.argmax(y_pred, axis=1)

pre_id=0
for x in y_pred:
	tar_pres=str(tar_pre[pre_id])+"\t"
	#print >>fout, tar_pre
	for y in x:
		tar_value="%.4f" % float(y)
		tar_pres=tar_pres + tar_value + "\t"
	print >>fout, pre_id,"\t",x[tar_pre[pre_id]],"\t",tar_pres
	pre_id=pre_id+1

del model
fout.close()
sys.exit()

