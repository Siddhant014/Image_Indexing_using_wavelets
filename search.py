# USAGE
# python search.py --index index.csv --query queries/103100.png --result-path dataset

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


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required = True,
	help = "Path to where the computed index will be stored")
ap.add_argument("-q", "--query", required = True,
	help = "Path to the query image")
ap.add_argument("-r", "--result-path", required = True,
	help = "Path to the result path")
args = vars(ap.parse_args())

# initialize the image descriptor
cd = ColorDescriptor((8, 12, 3))

# load the query image and describe it
query = cv2.imread(args["query"])
features = cd.describe(query)

# perform the search
searcher = Searcher(args["index"])
results = searcher.search(features)
# display the query
cv2.imshow("Query", query)
c=0
r=[]
# loop over the results
for (score, resultID) in results:
	# load the result image and display it
    result = cv2.imread(args["result_path"] + "/" + resultID)
    cv2.imshow("Result", result)
    r.append(result)
    cv2.waitKey(0)
os.chdir('C:\\Users\\Aaron Stone\\Desktop\\Minor Repository\\search-engine\\results')    
for i in r: 
    c=c+1
    cv2.imwrite(str(c)+".png",i)
c=c+1
cv2.imwrite(str(c)+".png",query)