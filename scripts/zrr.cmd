#!/bin/bash

set -x

python "%~dp0\..\zipRecursive.py" out_name=%2 root_dir=%3  file_pattern=%4 dir_pattern=%1 include_all=%5 postfix=8470p
