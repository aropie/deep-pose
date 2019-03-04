import numpy as np
import sys
sys.path.append('/home/marciniega/Things_Tools/scripts_python/math_tricks/')
import math_vect_tools as vcts

def check_list_of_atoms(res,atoms_list):
    atnames=''
    if atoms_list is None: # consider all the atoms
       atnames = res.atomnames
    elif type(atoms_list) is list: # check if a list is given
       atnames = atoms_list
    else:
      raise ListCheckError("The atoms_to_consider should be given as a list")
    return atnames

def compute_distance(ref,tar,atom_name):
    plot_data = []
    for i in range(len(tar)): # same size
        tar_res = tar[i]
        ref_res = ref[i]
        #print ref_res.CA.coord
        plot_data.append(Resi_plot(tar_res.resi,tar_res.resn,vcts.distance(getattr(ref_res,atom_name).coord, getattr(tar_res,atom_name).coord)))
    return plot_data

def get_dihedral_coord(resi,atom_list):
    vect = []
    for atm in atom_list:
        if hasattr(res, atm):
           vect.append(getattr(res, atm))
        else:
           break
    return vect

def compute_phipsi(structure):
    plot_data = []
    for i in range(1,len(structure)-1): # same size
        res_pre = structure[i-1]
        res = structure[i]
        res_nex = structure[i+1]
        phi = vcts.dihedral(getattr(res_pre,'C').coord,getattr(res,'N').coord,getattr(res,'CA').coord,getattr(res,'C').coord)
        psi = vcts.dihedral(getattr(res,'N').coord,getattr(res,'CA').coord,getattr(res,'C').coord,getattr(res_nex,'N').coord)
        plot_data.append(Resi_plot(res.resi,res.resn,[phi,psi]))
    return plot_data

def dihedral_diff(gr1,gr2):
    data_phi = range(len(gr1))
    data_psi = range(len(gr1))
    for val in range(len(gr1)):
        data_phi[val] = gr1[val][0] - gr2[val][0]
        data_psi[val] = gr1[val][1] - gr2[val][1]
    return data_phi, data_psi
