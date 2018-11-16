FLAGS64= -I/usr/include/python2.7 -I/usr/include/python2.7/numpy
FLAGS64_WIN=-I G:\Python27\include -I G:\Python27\Lib\site-packages\numpy\core\include
LIBS=-lpython2.7
LIBS_win=-L G:\Python27\libs -lpython27


all: IMGaffine.so APGLASSOup.so softresh.so

# ---- Link --------------------------- 
IMGaffine.so:  IMGaffine_c.o
	${CC} -shared $< -o $@ ${LIBS}
APGLASSOup.so:  APGLASSOup.o
	${CC} -shared $< -o $@ ${LIBS}	
softresh.so : softresh_c.o
	${CC} -shared $< -o $@ ${LIBS}
	
IMGaffine.pyd:  IMGaffine_c_w.o
	${CC} -shared $< -o $@ ${LIBS_win}
APGLASSOup.pyd:  APGLASSOup_w.o
	${CC} -shared $< -o $@ ${LIBS_win}	
softresh.pyd : softresh_c_w.o
	${CC} -shared $< -o $@ ${LIBS_win}
	
# ---- ${CC} C compile ------------------

IMGaffine_c.o:  IMGaffine_c.c C_arraytest.h
	${CC}  -c -fPIC $< ${FLAGS64} -o $@	
APGLASSOup.o:  APGLASSOup.c C_arraytest.h
	${CC}  -c -fPIC $< ${FLAGS64} -o $@
softresh_c.o: softresh_c.c softresh_arraytest.h
	${CC} -c -fPIC $< ${FLAGS64} -o $@
	
IMGaffine_c_w.o:  IMGaffine_c.c C_arraytest.h
	${CC}  -c -fPIC $< ${FLAGS64_WIN} -o $@	
APGLASSOup_w.o:  APGLASSOup.c C_arraytest.h
	${CC}  -c -fPIC $< ${FLAGS64_WIN} -o $@
softresh_c_w.o: softresh_c.c softresh_arraytest.h
	${CC} -c -fPIC $< ${FLAGS64_WIN} -o $@