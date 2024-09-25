import pyrosetta
pyrosetta.init('-mute all')
from pyrosetta.rosetta.core.scoring import ScoreType

import menten_gcn as mg
import menten_gcn.decorators as decs

from spektral.layers import ECCConv, GlobalSumPool
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler 
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model


import numpy as np
import pandas as pd
import random
import glob
import math
import argparse

def calculate_distance(res1, res2):
    coord1 = pose.residue(res1).xyz("CA")
    coord2 = pose.residue(res2).xyz("CA")

    #euclidean distance
    distance = math.sqrt((coord1[0] - coord2[0])**2 +
                         (coord1[1] - coord2[1])**2 +
                         (coord1[2] - coord2[2])**2)
    return distance

decorators = [decs.CBCB_dist(use_nm=True), decs.Rosetta_Ref2015_TwoBodyEneriges(individual=True, score_types=[ScoreType.fa_rep, ScoreType.fa_atr, ScoreType.fa_sol, ScoreType.lk_ball_wtd, ScoreType.fa_elec, ScoreType.hbond_sr_bb, ScoreType.hbond_lr_bb, ScoreType.hbond_bb_sc, ScoreType.hbond_sc])]
data_maker = mg.DataMaker(decorators=decorators,
                           edge_distance_cutoff_A=8.0,
                           max_residues=15,
                           nbr_distance_cutoff_A=10.0)

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input', type=str, required=True, help='input file (PDB format)')
parser.add_argument('-o', '--output', type=str, default='out.csv', help='output file name. default=out.csv')
args = parser.parse_args()

#TO DO: load the model here
pdb = args.input
pose = pyrosetta.pose_from_pdb(pdb)
wrapped_pose = mg.RosettaPoseWrapper(pose)
custom_objects = {'ECCConv': ECCConv, 'GlobalSumPool': GlobalSumPool}
model = load_model('disulfinet3d.keras', custom_objects)

pairs = []

for i in range(1, pose.size() + 1):
    for j in range(1, pose.size() +1):
        if j != i and j > i:
            distance = calculate_distance(i, j)
            if distance <= 10:
                pairs.append([i, j])

Xs_ = []
As_ = []
Es_ = []

for i in pairs:
    X, A, E, resids = data_maker.generate_input( wrapped_pose, i)
    Xs_.append(X)
    As_.append(A)
    Es_.append(E)

Xs_ = np.asarray(Xs_)
As_ = np.asarray(As_)
Es_ = np.asarray(Es_)

y_pred = model.predict([Xs_, As_, Es_])

df = pd.DataFrame({ 'disulfide': pairs, 'probability': y_pred.flatten()})
df.to_csv(args.output, index=False)