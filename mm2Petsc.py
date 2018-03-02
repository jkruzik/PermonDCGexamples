#!/usr/bin/env python3

import scipy.io
import os
import sys

dirs = os.environ['PETSC_DIR']
sys.path.insert(0, dirs+'/bin/pythonscripts/')
sys.path.insert(0, dirs+'/bin/')

import PetscBinaryIO

outdir = 'inputs'

def convert(filename):
    matName = filename.replace("/",".")
    matName= matName.split(".")[-2]
    outputfile = matName+'.mat'
    if outdir: outputfile = outdir+'/'+outputfile
    mfile = open(outputfile,'w')
    files = [filename]
    vecName = filename[:-4]+"_b"+filename[-4:]
    if os.path.isfile(vecName):
        files.append(vecName)

    for f in files:
        print(f)
        A = scipy.io.mmread(f)
        try:
            if A.shape[1]!=1:
                PetscBinaryIO.PetscBinaryIO().writeMatSciPy(mfile, A)
            else:
                PetscBinaryIO.PetscBinaryIO().writeVec(mfile, A)
        except:
            print('Error Creating file '+outputfile)
            if os.path.isfile(outputfile):
                os.remove(outputfile)
                print('File has been removed')
    
if __name__ == "__main__":
    if(len(sys.argv)!=2):
        print('Usage : python '+sys.argv[0]+' pathToMatrixFiles')
        exit(1)
    convert(sys.argv[1])

