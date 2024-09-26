import pyrosetta
pyrosetta.init('-mute all')
from pyrosetta.rosetta.core.scoring import ScoreType

import menten_gcn as mg
import menten_gcn.decorators as decs

from spektral.layers import *
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler 
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import random
import glob
import math


data = np.load('graphs.npz')
Xs = data['Xs']
As = data['As']
Es = data['Es']
outs = data['outs']

decorators = [decs.CBCB_dist(use_nm=True),
              decs.Rosetta_Ref2015_TwoBodyEneriges(individual=True, score_types=[ScoreType.fa_rep, ScoreType.fa_atr, ScoreType.fa_sol, ScoreType.lk_ball_wtd, ScoreType.fa_elec, ScoreType.hbond_sr_bb, ScoreType.hbond_lr_bb, ScoreType.hbond_bb_sc, ScoreType.hbond_sc])]
data_maker = mg.DataMaker(decorators=decorators,
                           edge_distance_cutoff_A=8.0,
                           max_residues=15,
                           nbr_distance_cutoff_A=10.0)

X_in, A_in, E_in = data_maker.generate_XAE_input_layers()

L1 = ECCConv(30, activation='relu')([X_in, A_in, E_in])
L1_drop = Dropout(0.3)(L1)
L2 = ECCConv(30, activation='relu')([L1_drop, A_in, E_in])
L2_drop = Dropout(0.3)(L2)
L3 = GlobalSumPool()(L2_drop)
L4 = Flatten()(L3)
output = Dense(1, name="out", activation="sigmoid")(L4)

model = Model(inputs=[X_in,A_in,E_in], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy' )
model.summary()

Xs = np.asarray( Xs )
As = np.asarray( As )
Es = np.asarray( Es )
outs = np.asarray( outs )

# Train Test split
X_train, X_val, A_train, A_val, E_train, E_val, y_train, y_val = train_test_split(Xs, As, Es, outs, test_size=0.2, random_state=42)

# Random Over Sampling of training split
ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
Xs_reshaped = X_train.reshape(X_train.shape[0], -1)
X_reshaped, y_ros = ros.fit_resample(Xs_reshaped, y_train)
num_features = X_train.shape[1:]
X_ros = X_reshaped.reshape(-1, *num_features)

As_reshaped = A_train.reshape(A_train.shape[0], -1)
A_reshaped, _ = ros.fit_resample(As_reshaped, y_train)
num_features = A_train.shape[1:]
A_ros = A_reshaped.reshape(-1, *num_features)

Es_reshaped = E_train.reshape(E_train.shape[0], -1)
E_reshaped, _ = ros.fit_resample(Es_reshaped, y_train)
num_features = E_train.shape[1:]
E_ros = E_reshaped.reshape(-1, *num_features)

history = model.fit(x=[X_ros, A_ros, E_ros], y=y_ros, batch_size=1000, epochs=100, validation_data=([X_val, A_val, E_val], y_val))
model.save("disulfinet3d-2.keras")
