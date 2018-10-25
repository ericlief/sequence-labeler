#!/bin/bash

# Tokenize
#for file in xaa xab xac xad xae xaf xag xah xai xaj xak xal xam; do
#    cat $file | python /home/liefe/mypkgs/mpkgs/preproc/preproc.py --out ${file}.tokenized
#done
file="$1"
echo $file
cat $file | python /home/liefe/mypkgs/mypkgs/preproc/preproc.py --out ${file}.tokenized


# Sort and randomize
#for file in xaa xab xac xad xae xaf xag xah xai xaj; do cat ${file}.tokenized; done | sort -u --output=xXX.sorted

# Just print to one file
#for file in xaa xab xac xad xae xaf xag xah xai xaj; do
#    cat ${file}.tokenized
#done > xXX.sorted

