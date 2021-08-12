# Copyright (c) Chris Choy (alex.kang@kuleuven.be).
#

import os
import sys
import logging
import decimal
import glob


def gene_filetxt(class_name):


    #bathtub bed bench bookshelf boottle chair cone cup curtain 
    # desk dresser flower_pot glass_box toilet stool sofa stairs table
    #car airplane person
    #guitat keyboard lamp mantel monitor night_stand piano plant range_hood 
    #sink sofa stairs stool table toilet tv_stand vase wardrobe xbox
    #fnames = sorted([os.path.relpath(fname, self.root) for fname in fnames])

    # os.path.basename(f)
    # os.path.dirname(f)
    # os.path.splitext(f)
    # os.path.splitext(os.path.basename(f))

    root = "/home/eavise/MinEngineDeve/ModelNet40"
    train_percentile = 0.80
    valid_percentile = 1-train_percentile
    readpath1 = os.path.join(root, class_name, "train/*.off")
    readpath2 = os.path.join(root, class_name, "test/*.off")
    #print(tpath)
    fnames = glob.glob(readpath1)
    fnames = sorted([os.path.relpath(fname, root) for fname in fnames])
    files = fnames
    assert len(files) > 0, "No file loaded"
    logging.info(
        f"Loading the train files from {root} with {len(files)} files"
    )
    counter = 0
    write_path1 =  os.path.join(root, "test_" + class_name + ".txt" )
    #print(write_path1)
    write_path2 =  os.path.join(root, "train_" + class_name + ".txt" )
    write_path3 =  os.path.join(root, "valid_" + class_name  + ".txt" )
    switch_file_flag = False
    # splitting the traing dataset into two part, one for traning, one for validation
    f = open(write_path2, 'w')    
    for fname in fnames:
        counter =counter+1
        if decimal.Decimal(counter) / decimal.Decimal(len(files)) <= train_percentile:
            f.write(fname)
            f.write('\n')
        elif (decimal.Decimal(counter) / decimal.Decimal(len(files)) > train_percentile) and (not switch_file_flag):
            f.close()
            f = open(write_path3, 'w')
            f.write(fname)
            f.write('\n')   
            switch_file_flag = True
        elif  (decimal.Decimal(counter) / decimal.Decimal(len(files)) > train_percentile) and (switch_file_flag):
            f.write(fname)
            f.write('\n')               
    f.close()

    fnames = glob.glob(readpath2)
    fnames = sorted([os.path.relpath(fname, root) for fname in fnames])
    files = fnames
    assert len(files) > 0, "No file loaded"
    logging.info(
        f"Loading the test files from {root} with {len(files)} files"
    )
    f = open(write_path1, 'w')    
    for fname in fnames:
        f.write(fname)
        f.write('\n')    
    f.close()


if __name__ == "__main__":
   gene_filetxt("desk")