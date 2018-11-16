#include <stdio.h>
#include <stdlib.h>
#include <XVVideo.h>
#include <XVMpeg.h>
#include <XVAVI.h>
#include "XVV4L2.h"
#include <XVDig1394.h>
#include "XVImageRGB.h"
//#include "XVImageIO.h"


//#include "XVImageBase.h"
//#include <XVImageScalar.h>

#define SRC_MPG 1
#define SRC_AVI 2
#define SRC_USB_CAM 3
#define SRC_DIG_CAM 4

#define IMG_FMT XV_RGB

#define NCHANNELS 3

#define DIG_DEV_NAME "/dev/fw1"
#define DIG_DEV_FORMAT "S0R0"

#define USB_DEV_NAME "/dev/video0"
#define USB_DEV_FORMAT "RGB3"

#define MPG_FNAME "/home/abhineet/G/UofA/Thesis/Xvision/XVision2/src/sdf2.mpg"
#define AVI_FNAME "/home/abhineet/G/UofA/Thesis/#Code/Videos/dl_mugI_s2.avi"

using namespace std;

typedef XVVideo<XVImageRGB<XV_RGB> > VID;
typedef XVMpeg<XVImageRGB<XV_RGB> > AVI;
typedef XVMpeg<XVImageRGB<XV_RGB> >  MPG;
typedef XVV4L2<XVImageRGB<XV_RGB> > V4L2;
typedef XVDig1394<XVImageRGB<XV_RGB> > DIG1394;

static PyObject* initSource(PyObject* self, PyObject* args);
static PyObject* getFrame(PyObject* self, PyObject* args);
static PyObject* getFrame2(PyObject* self, PyObject* args);

static VID *vid=NULL;
static PyArrayObject *img;
static int  img_width, img_height;

static int xv_nch=sizeof(XV_RGB);
static int xv_row_size;
static int current_frame;
static int n_buffers;