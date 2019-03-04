#!/usr/bin/env python
import sys
sys.path.append("/home/marciniega/Things_Dock/Proyecto/clean_start/pdbmani_adrian/pdb_manipulation")
sys.path.append("/home/marciniega/Things_Dock/Proyecto/clean_start/pdbmani_adrian/math_tricks")
from pdbqt import Pdbqt, PdbqtRes
from read_pdb_tools import Pdb
import cPickle as pickle
import math_vect_tools
import os

WORK_DIR = "/home/marciniega/Things_Dock/Proyecto/clean_start"
POSES_DIR = WORK_DIR+"/poses"
DOCK_DIR = WORK_DIR+"/docked_data"
REF_DIR = WORK_DIR+"/reference_data"
OBJ_DIR = WORK_DIR+"/working_objects"

def prepareBranches(lig, res):
    """
    Sets close branches to each branch
    """
    for bl in lig.branches:
        bl_center = bl.getGeometricCenter()
        for r in res.residues.values():
            for br in r.branches:
                if len(br.atoms) == 0: 
                   continue
                distance = math_vect_tools.distance(bl_center, br.getGeometricCenter())
                if distance <= RNG:
                    bl.close_branches.append((br, distance))
                    br.close_branches.append((bl, distance))

def detachPoses(lig_file):
    """
    Recives a docked ligand file a list of tuples (pose, score)
    where pose is a pdbqt object containing only the pose and
    score is the score asociated to that pose
    """
    pose_num = 1
    poses = []
    is_reading = False
    for line in lig_file:
        if line[:18].strip() == "REMARK VINA RESULT":
            score = line[20:32].strip()
        elif line[:4].strip() == "ROOT":
            is_reading = True
            f = open(str(pose_num) + ".pdbqt", "wb")
            continue
        elif line[:6].strip() == "ENDMDL":
            is_reading = False
            f.close()
            p = Pdbqt(f.name)
            poses.append((p, score))
            os.remove(f.name)
            pose_num += 1
            continue
        elif is_reading:
            f.write(line)
    return poses

def fileLen(fname):
    """
    Returns the number of lines in a file
    """
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

if __name__ == "__main__":

    RNG = 6
    # This dictionary is build by the get_brach_dict_pkl.py script
    DICTIONARY = pickle.load(open(OBJ_DIR+"/resi_branch_dict.pkl", "rb"))

    list_of_docked = sys.argv[1]
    
    for f in open(list_of_docked,'r').readlines():
        try:
            f = f.split('\n')[0]
            file_name = f[:-6]
            dir_name = POSES_DIR +'/'+ file_name
            os.mkdir(dir_name)
            docked_file = open(DOCKED_DIR + "/" + f , 'rb')
            ref_lig = Pdbqt(REF_DIR+"/ligands_pdbqt/L_" + f)
            res = PdbqtRes(REF_DIR+"/residues_pdbqt/R_" + f, DICTIONARY)
            poses = detachPoses(docked_file)
            docked_file.close()
            i = 1
            for (pose, score) in poses:
                prepareBranches(pose, res)
                rmsd = ref_lig.getRmsd(pose)
                pose_file = open(dir_name + "/" + str(i) + "_" + f, "wb")
                pose_file.write(str(rmsd) + "    " + str(score) + "\n")
                for br in pose.branches:
                    pose_file.write(br.info())
                pose_file.close()
                i += 1

        except Exception, err:
            sys.stderr.write(file_name + "   ")
            sys.stderr.write(str(err) + "\n")
            # Hasta aqui, ya dividimos al archivo de docking en un objeto pdbqt para
            # cada pose. La lista 'poses' tiene tuplas con el objeto pdbqt de la pose
            # y su score. Falta calcular el rmsdi de cada pose con respecto a la original
            # y hacer el archivo de cada pose con los pares. Falta igual acomodar los pares
            # en un formato de columnas.
