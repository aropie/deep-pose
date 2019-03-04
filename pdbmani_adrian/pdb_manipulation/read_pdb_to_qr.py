#!/usr/bin/env python
import sys
import numpy as np

def fix_atom(at_name):
    at_name = at_name.strip(' ')
    if len(at_name) == 4 :
       new_name = at_name[1:]+at_name[0]
       at_name = new_name
    return at_name

def fix_resn(rs_name):
    return rs_name.strip(' ')

def fix_resi(rs_name):
    return rs_name.strip(' ')

def sigma_to_radii(sigma):
    """Sigma value obtained from amber99sb as in gromacs"""
    return '%.4f'%(sigma*np.power(2,1./6)*5.)

def retain_atoms(pdbfile):
    dat = []
    for line in pdbfile:
        line = line.split('\n')[0]
        if line.split()[0] == "ATOM" or line.split()[0] == "HETATM":
           dat.append(line)
    return dat

def get_coordinates(in_line):
    return [float(in_line[30:38]),\
            float(in_line[38:46]),\
            float(in_line[46:54])]

def read_nonbonded(path_to_ff):
    dat = {}
    for line in open(path_to_ff+'/'+'ffnonbonded.itp','r')\
        .readlines():
        line = line.split('\n')[0].split()
        if len(line) == 7 and line[4] == 'A' :
           dat[line[0]] = line[5]+'_'+line[6]
    return dat

def read_restype(path_to_ff):
    dat = {}
    for line in open(path_to_ff+'/'+'residuetypes_for_qr.dat','r')\
        .readlines():
        line = line.split('\n')[0].split()
        dat[line[0]] = line[1]
    return dat

def get_resi_info(resfile,res_am):
    resi_data = open(resfile,'r').readlines()
    flag = False
    running_dic = {}
    for line in resi_data:
        line = line.split('\n')[0]
        if len(line) > 1 and line[0] == '[' and line.split()[1] == res_am:
           flag = True
           continue
        if flag and len(line.split()) > 1 and not len(line.split('['))>1:
           pdb_type = line.split()[0]
           amber_type = line.split()[1]
           amber_char = '%.4f'%float(line.split()[2])
           running_dic[pdb_type] = (amber_type,amber_char)
           continue
        if flag and line.split() and line.split()[1] == "bonds" :
           break
        if flag and len(line) > 1 and line[0] == '[':
           break
    return running_dic

def get_atomtype(pdb_res):
    resi_type = resitype_dic[pdb_res]
    if pdb_res == 'HIS':
       pdb_res = 'HIE'
    return get_resi_info(path_to_force+'/'+resi_type,pdb_res)

def get_atomrad(amber_atom_sig):
    return sigma_to_radii(amber_atom_sig)

def get_atomsig(amber_atom_type):
    return float(atomtype_dic[amber_atom_type].split('_')[0])

def get_atomeps(amber_atom_type):
    return float(atomtype_dic[amber_atom_type].split('_')[1])

def update_extrm(value,curr_0,curr_1):
    if value < curr_0:
       curr_0 = value
    if value > curr_1:
       curr_1 = value
    return [curr_0,curr_1]

def store_extrem(coords,curr):
    data = []
    if len(curr) == 0:
       data = [coords[0],coords[0],\
               coords[1],coords[1],\
               coords[2],coords[2]]
    else:
       temp = update_extrm(coords[0],curr[0],curr[1])
       data += temp
       temp = update_extrm(coords[1],curr[2],curr[3])
       data += temp
       temp = update_extrm(coords[2],curr[4],curr[5])
       data += temp
    return data

def base_length(val_min,val_max):
    return val_max-val_min

def grid_length():
    dim = []
    dim.append(base_length(curr[0],curr[1]))
    dim.append(base_length(curr[2],curr[3]))
    dim.append(base_length(curr[4],curr[5]))
    dim = np.array(dim)
    return dim+20,(dim+20)*1.333

def check_dime(cutoff=0.5):
    #c = 4 # for 4 processors
    c = 5 # for 5 processors
    grid_dimentions = []
    for val in fine_grid:
        grd_step = val
        cnt = 1
        while grd_step > cutoff:
              num = val
              dem = float(cnt*np.power(2,c+1)) + 1.0
              grd_step = num/dem
              cnt +=1
        grid_dimentions.append(int(dem))
    return grid_dimentions

def check_resn(rsd):
    """Find the right residue name in order to match the topology names.
       THERE IS A LOT TO DO HERE."""
    name = rsd.resn
    atomsnames = rsd.atomnames
    if name == "CYS" and len(atomsnames) == 10:
       name = "CYX"
    if name == "HIS" and len(atomsnames) == 18:
       name = "HIP"
    if name == "HIS" and len(atomsnames) == 17 and 'HD1' in atomsnames:
       name = "HID"
    return name

def check_if_complete(rsd,tempdic):
    if not len(rsd.atomnames)==len(tempdic.keys()):
       print "residue not regconized. Different number of atoms."
       print "Atoms in the topology file: %s"%len(tempdic.keys())
       rsd.PrintResSummary()
       sys.exit()

def apply_changes(rsd,tempdic,outfile,ext_val):

    def check_if_correct(aname,rname,temp_dic):
        try:
           temp_dic[aname]
        except KeyError:
           print " %s is not a proper topology atom name of %s "%(aname,rname)
           print temp_dic.keys(),len(temp_dic.keys())
           sys.exit()

    atomsnames = rsd.atomnames
    new_atomnames = []
    for atn in atomsnames:
        new_name = fix_atom(atn)
        check_if_correct(new_name,rsd.resn,tempdic)
        new_atomnames.append(new_name)
        atom_type = tempdic[new_name][0]
        atom_chrg = tempdic[new_name][1]
        atom_sig = get_atomsig(atom_type)
        atom_eps = get_atomeps(atom_type)
        atom_rad = get_atomrad(atom_sig)
        at = rsd.GetAtom(atn)
        at.UpDateValue( 'name' , new_name )
        at.UpDateValue( 'occup', float(atom_chrg) )
        at.UpDateValue( 'rfact', float(atom_rad) )
        at.UpDateValue( 'rfact_std', float(atom_eps) )
        outfile.write('%3s   %4s   %8.4f   %8.4f    %10.7f\n'%(rsd.resn,\
                                                              at.name,\
                                                              at.occup,\
                                                              at.rfact,\
                                                              at.rfact_std))
        ext_val = store_extrem(at.coord,ext_val)
    rsd.UpDateName('atomnames',new_atomnames)
    return ext_val

path_to_force = 'amber99sb-ildn.ff'
atomtype_dic = read_nonbonded(path_to_force)
resitype_dic = read_restype(path_to_force)
