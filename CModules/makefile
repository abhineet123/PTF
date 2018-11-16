all: xv dl
xv: i t s r rt pt pr ps prt	
i: xvInput.so
t: xvSSDTransPy.so
s: xvSSDSE2Py.so
r: xvSSDRotatePy.so
rt: xvSSDRTPy.so
pt: xvSSDPyramidTransPy.so
ps: xvSSDPyramidSE2Py.so
pr: xvSSDPyramidRotatePy.so
prt: xvSSDPyramidRTPy.so
dl: distanceUtils.so
dw: distanceUtils.pyd

LIBSCV= `pkg-config --libs opencv`
FLAGSCV= `pkg-config --cflags opencv`
FLAGS64=-I/include/XVision2 -I${PVM_ROOT}/include -I/usr/include/dc1394 -I/usr/include -I/usr/include/python2.7 -I/usr/include/python2.7/numpy
LIBS=-L/usr/X11R6/lib -L/lib -L/usr/lib -L/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu \
	-lXVTrack -lXVDevs -lXVCons -lXVSeg -lXVTools -lXVImages\
	-ljpeg -ltiff -L/usr/X11R6/lib64 -lXext -lX11 -lavformat -lavcodec -lavutil -lpthread -lippi -lippcc -lipps  \
	-lraw1394 -ldc1394 -lmpeg -lpython2.7
MI_LIBS=-lpython2.7
DSST_HEADERS = DSST/wrappers.h DSST/CorrelationVariance.h DSST/HOG.h DSST/Params.h DSST/sse.hpp
WARNING_FLAGS = -Wfatal-errors -Wno-write-strings -Wno-cpp
FLAGSCV_WIN=-I C:/OpenCV/build/include 
LIBSCV_WIN = -L C:\OpenCV\build\x86\vc10\bin -lopencv_calib3d2410 -lopencv_contrib2410 -lopencv_core2410 -lopencv_features2d2410 -lopencv_flann2410 \
 -lopencv_gpu2410 -lopencv_highgui2410 -lopencv_imgproc2410 -lopencv_legacy2410 -lopencv_ml2410 -lopencv_nonfree2410 -lopencv_objdetect2410 \
 -lopencv_ocl2410 -lopencv_photo2410 -lopencv_stitching2410 -lopencv_superres2410 -lopencv_video2410 -lopencv_videostab2410


# ---- distanceUtils --------------------------- 
distanceUtils.pyd:  distanceUtils.wo
	g++ -shared distanceUtils.wo  -o distanceUtils.pyd -L C:\Python27\libs -lpython27
distanceUtils.wo:  distanceUtils.c distanceUtils.h
	g++ -c -fPIC ${WARNING_FLAGS} distanceUtils.c -I C:/Python27/include -I C:/Python27/Lib/site-packages/numpy/core/include -msse2 -o distanceUtils.wo
	
distanceUtils.so:  distanceUtils.o
	${CXX} -shared distanceUtils.o -o distanceUtils.so -lpython2.7	${LIBSCV}
distanceUtils.o: distanceUtils.c distanceUtils.h ${DSST_HEADERS}
	${CXX} -c -fPIC -O3 ${WARNING_FLAGS} distanceUtils.c  -D ENABLE_CV -I/usr/include -I/usr/include/python2.7 -I/usr/include/python2.7/numpy ${FLAGSCV} -o  distanceUtils.o

# ---- xvInput --------------------------- 
xvInput.so:  xvInput.o
	${CXX} -shared xvInput.o -o xvInput.so ${LIBS}	
xvInput.o: xvInput.cc xvInput.h
	${CXX} -c -fPIC xvInput.cc ${FLAGS64} -o  xvInput.o

# ---- xvSSDRotatePy --------------------------- 
xvSSDRotatePy.so:  xvSSDRotatePy.o
	${CXX} -shared xvSSDRotatePy.o -o xvSSDRotatePy.so ${LIBS}	
xvSSDRotatePy.o: xvSSDRotatePy.cc xvSSDRotatePy.h
	${CXX} -c -fPIC xvSSDRotatePy.cc ${FLAGS64} -o  xvSSDRotatePy.o
	
# ---- xvSSDTransPy --------------------------- 
xvSSDTransPy.so:  xvSSDTransPy.o
	${CXX} -shared xvSSDTransPy.o -o xvSSDTransPy.so ${LIBS}
xvSSDTransPy.o: xvSSDTransPy.cc xvSSDTransPy.h
	${CXX} -c -fPIC xvSSDTransPy.cc ${FLAGS64} -o  xvSSDTransPy.o

# ---- xvSSDRTPy --------------------------- 
xvSSDRTPy.so:  xvSSDRTPy.o
	${CXX} -shared xvSSDRTPy.o -o xvSSDRTPy.so ${LIBS}	
xvSSDRTPy.o: xvSSDRTPy.cc xvSSDRTPy.h
	${CXX} -c -fPIC xvSSDRTPy.cc ${FLAGS64} -o  xvSSDRTPy.o
	
# ---- xvSSDSE2Py --------------------------- 
xvSSDSE2Py.so:  xvSSDSE2Py.o
	${CXX} -shared xvSSDSE2Py.o -o xvSSDSE2Py.so ${LIBS}
xvSSDSE2Py.o: xvSSDSE2Py.cc xvSSDSE2Py.h
	${CXX} -c -fPIC xvSSDSE2Py.cc ${FLAGS64} -o  xvSSDSE2Py.o
	
# ---- xvSSDPyramidTransPy --------------------------- 
xvSSDPyramidTransPy.so:  xvSSDPyramidTransPy.o
	${CXX} -shared xvSSDPyramidTransPy.o -o xvSSDPyramidTransPy.so ${LIBS}
xvSSDPyramidTransPy.o: xvSSDPyramidTransPy.cc xvSSDPyramidTransPy.h
	${CXX} -c -fPIC xvSSDPyramidTransPy.cc ${FLAGS64} -o  xvSSDPyramidTransPy.o
	
# ---- xvSSDPyramidSE2Py --------------------------- 
xvSSDPyramidSE2Py.so:  xvSSDPyramidSE2Py.o
	${CXX} -shared xvSSDPyramidSE2Py.o -o xvSSDPyramidSE2Py.so ${LIBS}
xvSSDPyramidSE2Py.o: xvSSDPyramidSE2Py.cc xvSSDPyramidSE2Py.h
	${CXX} -c -fPIC xvSSDPyramidSE2Py.cc ${FLAGS64} -o  xvSSDPyramidSE2Py.o
	
# ---- xvSSDPyramidRotatePy --------------------------- 
xvSSDPyramidRotatePy.so:  xvSSDPyramidRotatePy.o
	${CXX} -shared xvSSDPyramidRotatePy.o -o xvSSDPyramidRotatePy.so ${LIBS}
xvSSDPyramidRotatePy.o: xvSSDPyramidRotatePy.cc xvSSDPyramidRotatePy.h
	${CXX} -c -fPIC xvSSDPyramidRotatePy.cc ${FLAGS64} -o  xvSSDPyramidRotatePy.o
	
# ---- xvSSDPyramidRTPy --------------------------- 
xvSSDPyramidRTPy.so:  xvSSDPyramidRTPy.o
	${CXX} -shared xvSSDPyramidRTPy.o -o xvSSDPyramidRTPy.so ${LIBS}
xvSSDPyramidRTPy.o: xvSSDPyramidRTPy.cc xvSSDPyramidRTPy.h
	${CXX} -c -fPIC xvSSDPyramidRTPy.cc ${FLAGS64} -o  xvSSDPyramidRTPy.o
	