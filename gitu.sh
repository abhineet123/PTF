#!/bin/bash
git.exe add --all .
if [ "$#" -ne 1 ]; then
   git.exe commit
else
	if [ "$1" == "f" ]; then
		git.exe commit -m "minor fix"
	else
		git.exe commit -m "$1"
	fi
fi
git.exe push origin master