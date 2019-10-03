# import the necessary packages
from pyimagesearch.colordescriptor import ColorDescriptor
from pyimagesearch.searcher import Searcher
import argparse
import cv2
import glob
import os
from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from matplotlib import pyplot as plt
import haarPsi as hp

print("Similarity Score")
print("Result name\tScore without pre processing\tScore after pre processing")       
mypath='C:\\Users\\Aaron Stone\\Desktop\\Minor Repository\\search-engine\\resizedresults'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
  images[n] = cv2.imread( join(mypath,onlyfiles[n]))
c=0
query=images[-1]
for img in images: 
    c=c+100
    name="Result"+str(c)
    withoutpre=hp.haar_psi(np.asarray(query),np.asarray(img),0)[0]
    withpre=hp.haar_psi(np.asarray(query),np.asarray(img))[0]
    print(name,"\t\t",withoutpre,"\t",withpre)

