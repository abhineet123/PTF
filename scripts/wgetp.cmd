#!/bin/bash

set -x

python3 "%~dp0\..\wgetByParts.py" url=%1 size=%2 n_parts=%3 part_size=%4 out_name=%5
