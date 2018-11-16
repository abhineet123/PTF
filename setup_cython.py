from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

'''
set python2 as the default one if python3 is also present

SET VS90COMNTOOLS=%VS100COMNTOOLS%
python setup_cython.py build_ext --inplace
python ImageDirectoryTracking.py --esm "G:\UofA\Thesis\#Code\Datasets\Human\nl_bookI_s3\*.jpg" Results
'''

setup(
    name="Cython Tracker Library New",
    cmdclass={'build_ext': build_ext},
    include_dirs=[np.get_include()],
    ext_modules=[
        Extension("utility", ["utility.pyx"]),
        Extension("cython_trackers.utility", ["cython_trackers/utility.pyx"]),
        Extension("cython_trackers.imgtool", ["cython_trackers/imgtool.pyx"]),
        Extension("cython_trackers.ESMTracker", ["cython_trackers/ESMTracker.pyx"]),
        Extension("cython_trackers.BMICTracker", ["cython_trackers/BMICTracker.pyx"]),
        Extension("cython_trackers.NNTracker", ["cython_trackers/NNTracker.pyx"]),
        Extension("cython_trackers.PFTracker", ["cython_trackers/PFTracker.pyx"]),
        Extension("cython_trackers.DESMTracker", ["cython_trackers/DESMTracker.pyx"]),
        Extension("cython_trackers.DLKTracker", ["cython_trackers/DLKTracker.pyx"]),
        Extension("cython_trackers.DNNTracker", ["cython_trackers/DNNTracker.pyx"]),
        Extension("cython_trackers.CTracker", ["cython_trackers/CTracker.pyx"])
    ]
)
    
