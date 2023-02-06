#!/bin/bash
rm -rf pb && python train.py --saved_model_path=./pb --tf

PACKAGES="100 500 1000 5000"
for p in $PACKAGES
do
  mkdir -p structured/$p
  rm -rf structured/$p/pb && python train.py --structured 1 --saved_model_path=./structured/$p/pb --item_packsize=$p  --tf
done
