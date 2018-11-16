#!/bin/bash -v

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
python "$DIR/../visualizeWithMotion.py" src_path=$1

