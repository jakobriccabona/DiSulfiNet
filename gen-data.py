import pyrosetta
pyrosetta.init('-mute all')
from pyrosetta.rosetta.core.scoring import ScoreType

import menten_gcn as mg
import menten_gcn.decorators as decs

from spektral.layers import *
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

decorators = [decs.CBCB_dist(use_nm=True), decs.Rosetta_Ref2015_TwoBodyEneriges(individual=True, score_types=[ScoreType.fa_rep, ScoreType.fa_atr, ScoreType.fa_sol, ScoreType.lk_ball_wtd, ScoreType.fa_elec, ScoreType.hbond_sr_bb, ScoreType.hbond_lr_bb, ScoreType.hbond_bb_sc, ScoreType.hbond_sc])]
data_maker = mg.DataMaker(decorators=decorators,
                           edge_distance_cutoff_A=8.0,
                           max_residues=15,
                           nbr_distance_cutoff_A=10.0)
data_maker.summary()

def get_cys(pose):
    disulfides = []
    for i in range(1, pose.size()+1):
        if pose.residue(i).name() == "CYS:disulfide":
            res = pose.residue(i)
            disulfide_partner = res.residue_connection_partner(res.n_current_residue_connections())
            if disulfide_partner > i:
                disulfides.append([i, disulfide_partner])
    return disulfides

def get_negative_examples(pose):
    negative_examples = []
    pairs = []
    disulfides = get_cys(pose)
    disulfide_res = {i for bond in disulfides for i in bond}

    for i in range(1, pose.size() + 1):
        if i not in disulfide_res:
            negative_examples.append(i)

    for i in negative_examples:
        for j in negative_examples:
            if i != j and j > i:
                distance = calculate_distance(i, j)
                if distance <= 10:
                    pairs.append([i, j])

    return pairs

def calculate_distance(res1, res2):
    coord1 = pose.residue(res1).xyz("CA")
    coord2 = pose.residue(res2).xyz("CA")
    
    # Calculate the Euclidean distance between the two coordinates
    distance = math.sqrt((coord1[0] - coord2[0])**2 + 
                         (coord1[1] - coord2[1])**2 + 
                         (coord1[2] - coord2[2])**2)
    return distance


pdb_list = glob.glob("/home/iwe14/Documents/database/cath_S40/*?.pdb")
Xs = []
As = []
Es = []
outs = []

for pdb_file in pdb_list:
    pose = pyrosetta.pose_from_pdb(pdb_file)
    cys = get_cys(pose)
    no_cys = get_negative_examples(pose)
    wrapped_pose = mg.RosettaPoseWrapper(pose)

    for i in cys:
        X, A, E, resids = data_maker.generate_input( wrapped_pose, i)
        Xs.append(X)
        As.append(A)
        Es.append(E)
        outs.append([1.0,])

    for i in no_cys:
        X, A, E, resids = data_maker.generate_input( wrapped_pose, i)
        Xs.append(X)
        As.append(A)
        Es.append(E)
        outs.append([0.0,])

Xs_np = np.array(Xs)
As_np = np.array(As)
Es_np = np.array(Es)
outs_np = np.array(outs)

np.savez_compressed('graphs.npz', Xs=Xs_np, As=As_np, Es=Es_np, outs=outs_np)
