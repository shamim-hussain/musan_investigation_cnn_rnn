

import numpy as np
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from pathlib import Path
from keras import backend as K
from keras.utils import to_categorical
import tensorflow as tf
from custom_layers import *
from keras.callbacks import *
from sklearn.utils import shuffle
from multiprocessing.pool import ThreadPool
from math import ceil
import sys
import h5py
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from plot_cm import plot_cm
from my_models import get_mobile_net as get_model
import inspect

codes=inspect.getsource(inspect.getmodule(inspect.currentframe()))

np.random.seed(7)
root_path = r'e:\musan_data_derived.h5'    #Path to the derived features

dtime = datetime.now().strftime('-%B-%d(%a)-%H-%H-%S')
fname = Path(sys.argv[0]).stem

train_split = .65
test_split = .75
modelname = 'WaveNetv8-'
num_feat = 20
seg_len = 200
feat = 'mfcc'
dtype = np.float32
bsize = 128
num_epochs=120

resultf = './results/'+modelname+str(seg_len)+dtime+'.npz'
modelf = './models/'+modelname+str(seg_len)+dtime+'.h5'
modelff = './models/'+modelname+str(seg_len)+dtime+'_final.h5'

ecatg = dict((c,i) for (i,c) in enumerate(['noise', 'music', 'speech']))

if (not 'X_train' in locals()) or input('Reload Data? [Y/N] :').lower()=='y':
    with h5py.File(root_path, mode='r') as db:
        fdict = dict((c, []) for c in ecatg)
        for k in db.keys():
            c = k.split('\\')[0]
            if c in ecatg:
                fdict[c].append(k)
        
        train, val, test = {}, {}, {}
        for k in fdict:
            np.random.shuffle(fdict[k])
            ut = int(len(fdict[k])*train_split)
            uv = int(len(fdict[k])*test_split)
            train[k], val[k], test[k] = fdict[k][:ut], fdict[k][ut:uv],\
                                    fdict[k][uv:]
        
        def frm_proc(frms):
            #frms = (frms-frms.mean(axis=(1,2), keepdims=True))/(np.amax(np.abs(frms), axis=(1,2), keepdims=True)+1e-2)#/frms.std(axis=(1,2), keepdims=True)
            return frms.astype(dtype)#
        
        def frm_gen(filtered):
            if len(filtered)<seg_len:
                filtered = np.pad(filtered, pad_width=((seg_len-len(filtered), 0), (0,0)), 
                                  mode='wrap')
            seg_points1 = np.arange(seg_len, len(filtered), seg_len)
            seg_points2 = np.arange(seg_len//2, len(filtered), seg_len)

            frms = np.stack(np.split(filtered, seg_points1)[:-1]+
                                    np.split(filtered, seg_points2)[1:-1]+
                                    [filtered[-seg_len:]])
            return frm_proc(frms)
        
        class data_gen:
            def __init__(self, dic):
                self.dic = dic
                self.labels = []
                
            def yield_dat(self):
                for k, files in self.dic.items():
                    ln = len(files)
                    print('Concatenating {} {} files...'.format(ln, k.upper()))
                    for i, fl in enumerate(files):
                        if not i % 50:
                            print('Read', str(i), 'out of', str(ln), 'files...')
                        
                        dat = frm_gen(db[fl][feat][:])
                        lbl = len(dat)*[float(ecatg[k])]
                        self.labels.append(lbl)
                        yield dat

        print('\nConcatenating train data...')
        dg = data_gen(train)
        if 'X_train' in locals():
            del X_train
        X_train = np.vstack(dg.yield_dat())
        Y_train = np.hstack(dg.labels)
        
        print('\nConcatenating validation data...')
        dg = data_gen(val)
        if 'X_val' in locals():
            del X_val
        X_val = np.vstack(dg.yield_dat())
        Y_val = np.hstack(dg.labels)
        
        print('\nConcatenating test data...')    
        dg = data_gen(test)
        if 'X_test' in locals():
            del X_test
        X_test = np.vstack(dg.yield_dat())
        Y_test = np.hstack(dg.labels)
        

K.clear_session()

in_shape = X_train.shape[1:]
model = get_model(in_shape, name=modelname)
print(model.summary())

lr0=5e-4
opt = Adam(lr=lr0)
model.compile(opt, 'sparse_categorical_crossentropy', ['acc'])

mchk = ModelCheckpoint(modelf, save_best_only='True', period=1,
                       verbose=1)

lr0=5e-4
opt = Adam(lr=lr0)
model.compile(opt, 'sparse_categorical_crossentropy', ['acc'])

class WarmStart(Callback):
    def __init__(self, lr_min=1e-5,lr_max=5e-4, T_0=1, M=2,
                 X_test=X_test, Y_test = Y_test, codes=codes):
        super().__init__()
        self.lr_min,self.lr_max,self.T_0, self.M=lr_min,lr_max,T_0, M
        self.S_i=0
        self.T_i=T_0
        self.X_test=X_test
        self.Y_test = Y_test
        self.codes = codes
    
    def on_epoch_begin(self,epoch, lr=0):
        ep=epoch
        print('LRS at :', epoch)
        if ep > self.S_i+self.T_i:
            print('==WARM RESTART==')
            self.S_i=ep
            self.T_i*=2
            #self.take_snap(epoch)
        print('Next restart at :', self.S_i+self.T_i+1)
            
        T_cur = ep-self.S_i
        lr_i = self.lr_min + 0.5*(self.lr_max-self.lr_min)*(1+
                                 np.cos(np.pi * (T_cur/self.T_i)))
        K.set_value(self.model.optimizer.lr, lr_i)
        print('Set learning rate to:', lr_i)
    
    def take_snap(self,ep):
        xstr='snapshot_ep'+str(ep)+'_'
        model.save(modelf(xstr))
        Yp_test = model.predict(self.X_test, batch_size=256, verbose=1)
        np.savez(resultf(xstr), Y_test=self.Y_test, Yp_test=Yp_test, 
                 codes=self.codes)
        print('SNAPSHOT TAKEN!')


ws = WarmStart()
fhist = model.fit(X_train, Y_train, batch_size=bsize, epochs=num_epochs, 
                  validation_data=[X_val, Y_val],
                  callbacks=[ws, mchk])

model.save(modelff)
Yp_val = model.predict(X_val, verbose=1, batch_size=256)
Yp_test = model.predict(X_test, verbose=1, batch_size=256)
np.savez(resultf, Y_val=Y_val, Yp_val=Yp_val, 
         Y_test=Y_test, Yp_test=Yp_test, 
         fhist=fhist.history, ecatg=ecatg, codes = codes,
         train=train, val=val, test=test)

ll_val = log_loss(Y_val, Yp_val)
ll_test = log_loss(Y_test, Yp_test)

acc_val = accuracy_score(Y_val, Yp_val.argmax(-1))
acc_test = accuracy_score(Y_test, Yp_test.argmax(-1))

cm_val = confusion_matrix(Y_val, Yp_val.argmax(-1))
cm_test = confusion_matrix(Y_test, Yp_test.argmax(-1))

plot_cm(cm_val, list(ecatg.keys()), True, 'Confusion Matrix - Validation')
plot_cm(cm_test, list(ecatg.keys()), True, 'Confusion Matrix - Test')

sns = np.vectorize(lambda x: 1 if x==ecatg['speech'] else 0)
acc_sns_val = accuracy_score(sns(Y_val), sns(Yp_val.argmax(-1)))
acc_sns_test = accuracy_score(sns(Y_test), sns(Yp_test.argmax(-1)))

print('Test Accuracy = {:0.4}%'.format(acc_test*100))











