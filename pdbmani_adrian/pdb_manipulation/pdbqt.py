import read_pdb_tools as rpt
from read_pdb_tools import Atom
import numpy as np
from numpy.linalg import norm
import subprocess
import os

class Branch(object):
    def __init__(self):
        self.atoms = []
        self.close_branches = []
        self.smi = ""
        
    def __str__(self):
        s = ""
        for a in self.atoms:
            s += str(a)
            s += "\n"
        return s

    def info(self):
        s = self.smi
        s += "\n"
        for b in self.close_branches:
            s += "%-15.13f"%b[1] + ", " + b[0].smi
            s+= "\n"
        return s

    def addAtom(self, atom):
        self.atoms.append(atom)
        
    def getGeometricCenter(self):
        """
        Returns the coords of the Residue's geometric center.
        """
        x, y, z = 0, 0, 0
        for atom in self.atoms:
                coord = atom.coord
                x += coord[0]
                y += coord[1]
                z += coord[2]
        l = len(self.atoms)
        return np.array([x/l, y/l, z/l])

    def getAtomCount(self):
        return len(self.atoms)
            
    def getSmi(self):
        h = str(hash(self)) + ".pdbqt"
        with open(h, 'w') as f:
            f.write(str(self))
        cmd = "obabel -ipdbqt " + h + " -ocan 2> /dev/null"
        result = subprocess.check_output(cmd, shell=True)
        os.remove(h)
        return str(result).split("\t")[0].lstrip()



class Pdbqt(rpt.Pdb):
    
    def __init__(self, name, ligandName="", l_chain=""):
        rpt.Pdb.__init__(self, name, ligandName, l_chain)
        self.branches = []
        self.atoms = []

        branch = Branch()
        with open(name, 'r') as f:
            for line in f:
                if line[:4] == 'ATOM' or line[:6] == "HETATM":
                    (atom_number, name, res_name, chain, res_id, coords, occup, rfact, element, charge) = self.splitInfo(line)
                    atom = Atom(atom_number, name, res_name, chain, res_id, coords, occup, rfact, element, charge)
                    
                    if line[:6] == "HETATM" and res_name != "HOH" and res_name != "CD":
                        atom.isHetatm = True
                        
                    branch.addAtom(atom)
                    self.atoms.append(atom)
                elif line[:6] == "BRANCH":
                    self.branches.append(branch)
                    branch.smi = branch.getSmi()
                    branch = Branch() 

            self.branches.append(branch)
            branch.smi = branch.getSmi()
            
                    
    def printInfo(self):
        print self.name
        print str(len(self.atoms)) + " atoms"
        print str(len(self.branches)) + " branches"
        print "___BRANCHES___"
        for smi, branch in self.branches.iteritems():
            print smi
            branch.printInfo()

    def getSmi(self, branch):
        h = str(hash(branch)) + ".pdbqt"
        with open(h, 'w') as f:
            f.write(str(branch))
        cmd = "obabel -ipdbqt " + h + " -ocan"
        result = subprocess.check_output(cmd, shell=True)
        os.remove(h)
        return str(result).split("\t")[0].lstrip()

    def getAtomCount(self):
        return len(self.atoms)

    def getBranchCount(self):
        return len(self.branches)

    def getRmsd(self, lig):
        add = 0
        for i in range(len(self.branches)):
            d = self.branches[i].getGeometricCenter() - lig.branches[i].getGeometricCenter()
            add += norm(d)
        l = len(self.branches)
        self.rmsd = np.sqrt(add/l)
        return self.rmsd

class PdbqtRes(Pdbqt):

    def __init__(self, name, dictionary, ligandName="", l_chain=""):
        rpt.Pdb.__init__(self, name, ligandName, l_chain)

        for res in self.residues.values():
            res.branches = []
            for i in range(len(dictionary[res.name])):
                res.branches.append(Branch());
            for atom in res.atoms:
                for br in dictionary[res.name]:
                    if atom.name in br['atms']:
                        index = dictionary[res.name].index(br)
                        res.branches[index].smi = dictionary[res.name][index]["smi"]
                        res.branches[index].addAtom(atom)
                        break

    def printBranches(self):
        for res in self.residues.values():
            print res
            for br in res.branches:
                print br.smi
                print br
