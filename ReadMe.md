Welcome to the home of **Python Tracking Framework (PTF)** - a collection of registration based trackers (like IC, ESM, NN, PF and L1) and related utilities implemented in Python and Cython along with supporting Matlab scripts

Prerequisites:
==============
* [OpenCV](http://opencv.org/)
* [Numpy](http://www.numpy.org/)
* [Scipy](https://www.scipy.org/)
* [Matplotlib](http://matplotlib.org/)
* [FLANN](http://www.cs.ubc.ca/research/flann/) (optional)
* [Cython](http://cython.org/) (optional)
* [MTF](http://webdocs.cs.ualberta.ca/~vis/mtf/) (optional)
* [Xvision](https://bitbucket.org/abhineet123/xvision2) (optional)
		
Installation
============

All optional Cython and C modules used by PTF can be compiled and installed by simply calling `make` from the root folder. This in turn calls the make commands in the following sub folders: `CModules`, `cython_trackers`, `l1`. 

* It also executes the command to compile and install the Python interface to the [**Modular Tracking Framework**](http://webdocs.cs.ualberta.ca/~vis/mtf/) called `pyMTF`. This requires the source code of this library to be present in `~/mtf` folder.  Change the variable `MTF_DIR` in the makefile if the source code is present elsewhere.  
* If Xvision is not installed, either remove `xv` from the `all` target of the makefile before calling the `make` command or call the following separate commands instead:

    * `make dl`
    * `make cython`
    * `make l1`
    * `make mtf`
	
* If make does not work on your system (e.g. if it has Windows OS), run `python setup_cython.py build_ext --inplace` to compile the Cython modules.	


Basic Usage
===========
Setting parameters:

Set `db_root_path` in `main.py` if running on a dataset sequence otherwise select `usb camera` for `source` and adjust `camera_id` in `main.py`. All other parameters can be  adjusted either from the GUI or in `config.py`.

Running with GUI:
```
python main.py
```
Running without GUI with default parameter settings in `config.py`:
```
python main.py 0
```
