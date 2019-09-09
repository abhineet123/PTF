#!/bin/bash

set -x

python2 "%~dp0\..\wgetByParts.py" url=$1 n_parts=$2 part_size=$3 size=$4 out_name=$5
