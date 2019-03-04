#!/usr/bin/env python
import sys
import cPickle as pickle
import os

LIST_DIR = "/home/marciniega/Things_Dock/Proyecto/clean_start/working_lists"

def split_info(line):
    r_name = line[:3]
    smi = line[7:25].strip()
    atms = line[26:67].strip().split(",")
    atm_type = line[67:].strip().split(",")
    return (r_name, smi, atms, atm_type)

def createDictionary(dic):
    dictionary = {}
    with open(dic, 'r') as d:
        for line in d:
            r_name, smi, atms, atm_types = split_info(line)
            if r_name not in dictionary.keys():
                dictionary[r_name] = []
            dictionary[r_name].append(({
                "smi": smi,
                "atm_types": atm_types,
                "atms": atms
                }))
    pickle.dump(dictionary, open("resi_branch_dict.pkl", "wb"))

createDictionary('LIST_DIR/filtered_branches_dict_list.txt')
