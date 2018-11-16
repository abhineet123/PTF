.PHONY: l1

all: xv dl cython l1 mtf

MTF_DIR ?= ~/mtf

xv:
	$(MAKE) xv -C ./CModules --no-print-directory
dl:
	$(MAKE) dl -C ./CModules --no-print-directory

cython:
	python setup_cython.py build_ext --inplace

mtf:
	$(MAKE) mtfp  -C ${MTF_DIR}  --no-print-directory
	
l1:
	$(MAKE) -C ./l1 --no-print-directory
		