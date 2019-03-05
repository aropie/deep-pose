#!/usr/bin/env python
import sys
from pdbqt import Pdbqt, PdbqtRes
import cPickle as pickle
import math_vect_tools
import os
sys.path.append("/home/marciniega/Things_Dock/Proyecto/clean_start/pdbmani_adrian/pdb_manipulation")
sys.path.append("/home/marciniega/Things_Dock/Proyecto/clean_start/pdbmani_adrian/math_tricks")

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
    Recieves a docking result file and divides it into pdb object
    per pose

    :param lig_file: docking result file
    :return: [(pose1, score1), (pose2, score2), ...]
    :rtype: (Pdbqt, int)[]

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

    for f in open(list_of_docked, "r").readlines():
        try:
            f = f.split("\n")[0]
            file_name = f[:-6]
            dir_name = POSES_DIR + "/" + file_name
            os.mkdir(dir_name)
            with open(os.path.join(DOCK_DIR, f), "rb") as docked_file:
                ref_lig = Pdbqt(REF_DIR+"/ligands_pdbqt/L_" + f)
                res = PdbqtRes(REF_DIR+"/residues_pdbqt/R_" + f, DICTIONARY)
                poses = detachPoses(docked_file)
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
