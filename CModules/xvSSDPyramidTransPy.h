#include <Python.h>
#include <numpy/arrayobject.h>

#include <XVSSD.h>
#include <XVTracker.h>
#include <XVMpeg.h>
#include <XVDig1394.h>
#include <XVV4L2.h>
#include <XVWindowX.h>

#include <iostream>
#include <vector>

#define XV_RED 0
#define XV_GREEN 1
#define XV_BLUE 2

#define CV_RED 0
#define CV_GREEN 1
#define CV_BLUE 2

#define NCHANNELS 3
#define NCORNERS 4

using namespace std;

#define PIX_TYPE XV_RGB24
#define PIX_TYPE_WIN XV_RGB


typedef XVImageRGB<PIX_TYPE> IMAGE_TYPE;
typedef XVVideo< IMAGE_TYPE > VID;
typedef XVInteractWindowX< PIX_TYPE_WIN > WIN_INT;

typedef XVPyramidStepper<XVTransStepper< IMAGE_TYPE > >  STEPPER_TYPE;

typedef XVSSD< IMAGE_TYPE, STEPPER_TYPE > TRACKER_TYPE;
typedef TRACKER_TYPE::SP STATE_PAIR_TYPE;
typedef STEPPER_TYPE::STATE_TYPE STATE_TYPE;

struct TrackerStruct {
	TRACKER_TYPE *ssd;
	STEPPER_TYPE *stepper;
	XVSize *xv_size;
	int current_frame;
	int show_xv_window;
	int steps_per_frame;
	int direct_capture;
	int no_of_levels;
	double scale;
	TrackerStruct(){
		current_frame=0;
		show_xv_window=0;
		steps_per_frame=1;
		direct_capture=1;
		no_of_levels=2;
		scale=0.5;
	}
};

static int no_of_trackers=0;
static int current_tracker_id=0;

static PyObject* initialize(PyObject* self, PyObject* args);
static PyObject* update(PyObject* self, PyObject* args);

static int  img_width, img_height;
static int xv_nch=sizeof(XV_RGB);
static int xv_line_width=3;
static IMAGE_TYPE* xv_frame;
static char* xv_data;
static int n_buffers=0;

static PyArrayObject *in_img_py;
static PyArrayObject *corners_array;
static XVPosition corners[NCORNERS];
static int* corners_data;

static WIN_INT *win_int=NULL;

static vector<TrackerStruct*> trackers;
static TrackerStruct* current_tracker;
static VID *vid=NULL;

static int first_win_id=0;
static int last_win_id=0;

static inline void numpyToXV() {
	//printf("in xvSSDTrans: numpyToXV with img_height=%d img_width=%d\n", img_height, img_width);
	char* np_data=(char*)in_img_py->data;
	int xv_location=0, np_location=0;
	for(int row=0; row<img_height; row++) {
		for(int col=0; col<img_width; col++) {
			for(int ch=0; ch<NCHANNELS; ch++) {
				//printf("%d ", (int)np_data[np_location]);
				xv_data[xv_location]=np_data[np_location];
				np_location+=in_img_py->strides[2];
				xv_location++;
			}
			//printf("\t");
		}
		//printf("\n------------------------------\n");
	}
}

static void updateCorners(STATE_PAIR_TYPE& currentState, float scale=1.0) {

	XV2Vec<double> points[NCORNERS];
	XV2Vec<double> tmpPoint = XV2Vec<double>(current_tracker->xv_size->Width() / 2,
		current_tracker->xv_size->Height() / 2);

	points[0] = - tmpPoint;
	points[1] = XV2Vec<double>(tmpPoint.PosX(), - tmpPoint.PosY());
	points[2] = tmpPoint;
	points[3] = XV2Vec<double>(- tmpPoint.PosX(), tmpPoint.PosY());

	for(int i=0; i<NCORNERS; ++i)
		corners[i] = points[i] + currentState.state;

	for(int i=0; i<4; i++) corners[i].setX((int)(corners[i].x()/scale)),
		corners[i].setY((int)(corners[i].y()/scale));
}