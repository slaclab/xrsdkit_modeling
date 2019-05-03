#!/bin/bash

badnm=31
goodnm=29
#badnm=34
#goodnm=30
#badnm=37
#goodnm=31

for loopidx in 0 1 2 3 4; do

#echo $badnm
#echo $goodnm
#echo $loopidx
mv 'gridsearch'$badnm'_'$loopidx'.yml' 'gridsearch'$goodnm'_'$loopidx'.yml'

done
