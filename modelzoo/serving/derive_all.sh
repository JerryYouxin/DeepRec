#!/bin/bash
ALL=`ls`
for m in $ALL
do
  cd $m; ./derive_pb.sh; cd ..
done
