#!/bin/bash

set -x

python3 ~/PTF/wgetByParts.py url=$1 n_parts=$2 start_id=$3 part_size=$4 size=$5 out_name=$6
