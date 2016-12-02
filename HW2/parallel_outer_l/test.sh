#!/bin/bash

for((i=$1;i<=$2;i+=1))
do
	echo 'Number of threads: ' $i
	for((j=1;j<=$3;j+=1))
	do
		echo 'Test iteration ' $j ' of ' $3
		./parallel_convolutions $i $4 ../data/input.pgm ./output.pgm >> output.log
	done
done
