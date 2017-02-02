#!/bin/bash


if [ $# -ne 2 ]; then
	echo "usage: ./compress_data.sh /path/to/datadir number_of_tars"
	exit 0 
fi

dirname=$1
num=$2

rank=1
while [ $rank -le $num ]
do
	cd $dirname
	cp features.mat_$rank features.mat
	cp labels.mat_$rank labels.mat
	tar cvf aws_data_$rank.tar features.mat labels.mat
	(( rank++ ))
done
