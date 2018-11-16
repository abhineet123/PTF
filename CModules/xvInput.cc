#include <Python.h>
#include <numpy/arrayobject.h>
#include "xvInput.h"

/* ==== Set up the methods table ====================== */
static PyMethodDef xvInputMethods[] = {
    {"initSource", initSource, METH_VARARGS},
	{"getFrame", getFrame, METH_VARARGS},
	{"getFrame2", getFrame2, METH_VARARGS},
    {NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the C_test functions ====================== */
// Module name must be xvInput in compile and linked
PyMODINIT_FUNC initxvInput()  {
    (void) Py_InitModule("xvInput", xvInputMethods);
    import_array();  // Must be present for NumPy.  Called first after above line.
}

static PyObject* initSource(PyObject* self, PyObject* args) {

	fprintf(stdout, "Starting initSource\n");

	//VID *vid=NULL;
	//FILE *fp;

	char* dig_dev_name=(char*)DIG_DEV_NAME;
	char *dig_dev_format=(char*)DIG_DEV_FORMAT;

	char* usb_dev_name=(char*)USB_DEV_NAME;
	char *usb_dev_format=(char*)USB_DEV_FORMAT;

	char* mpg_fname=(char*)MPG_FNAME;
	char *avi_fname=(char*)AVI_FNAME;
	
	char *dev_name, *dev_fmt;
	int img_source=SRC_MPG;

    /*parse first input array*/
    if (!PyArg_ParseTuple(args, "izz", &img_source, &dev_name, &dev_fmt)){
		fprintf(stdout, "\n----initSource: Input arguments could not be parsed----\n\n");
		return NULL;
	}
    fprintf(stdout, "img_source=%d\n", img_source);
	if (img_source == NULL){
		fprintf(stdout, "\n----initSource: img_source is NULL----\n\n");
		return NULL;
	}

    switch(img_source) {
    case SRC_MPG: {
        if (dev_name!=NULL) {
            mpg_fname=dev_name;
        }
		fprintf(stdout, "Opening mpeg file %s\n", mpg_fname);
        vid = new MPG(mpg_fname);
        break;
    }
    case SRC_AVI: {
        if (dev_name!=NULL) {
            avi_fname=dev_name;
        }
		fprintf(stdout, "Opening avi file %s\n", avi_fname);
        vid = new AVI(avi_fname);
        break;
    }
    case SRC_USB_CAM: {
        if (dev_name!=NULL) {
            usb_dev_name=dev_name;
        }
        if (dev_fmt!=NULL) {
            usb_dev_format=dev_fmt;
        }
		fprintf(stdout, "Opening USB camera %s\n", usb_dev_name);
        vid = new V4L2(usb_dev_name, usb_dev_format);
        break;
    }
	case SRC_DIG_CAM: {
		if (dev_name!=NULL) {
			dig_dev_name=dev_name;
		}
		if (dev_fmt!=NULL) {
			dig_dev_format=dev_fmt;
		}
		fprintf(stdout, "Opening FIREWIRE camera %s\n", dig_dev_name);
		vid = new DIG1394(DC_DEVICE_NAME, dig_dev_format, DIG1394_NTH_CAMERA(0));
		break;
					  }
    default: {
        fprintf(stdout, "Invalid image source provided\n");
        exit(0);
    }
    }

	FILE *fp=fopen("xvInput.bin", "wb");
	if (fp==NULL){
		fprintf(stdout, "File xvInput.bin could not be opened for writing\n");
		return NULL;
	}
	fwrite(&vid, sizeof(vid), 1, fp);
	fclose(fp);

    n_buffers = vid->buffer_count();
    
    vid->initiate_acquire(current_frame);
    printf("Done initiate_acquire\n");

    vid->wait_for_completion(current_frame);
    printf("Done wait_for_completion\n");

    XVImageRGB<IMG_FMT> & xv_frame=vid->frame(current_frame);
    printf("Done accessing frame\n");

    img_height=xv_frame.SizeY();
    img_width=xv_frame.SizeX();

	int dims[3];
	dims[0]=img_height;
	dims[1]=img_width;
	dims[2]=3;

	img=(PyArrayObject *) PyArray_FromDims(3,dims,NPY_UINT8);
	printf("initSource: img=%d\n", img);


	xv_row_size=img_width*xv_nch;

	//fprintf(stdout, "vid=%d\n", vid);

	//if((fp=fopen("/home/abhineet/vid.bin", "wb"))==NULL) {
	//	printf("Cannot open file.\n");
	//	return NULL;
	//}
	//fwrite(vid, sizeof(VID), 1, fp);
	//fclose(fp);

    return Py_BuildValue("ii", img_width, img_height);
}

static PyObject* getFrame(PyObject* self, PyObject* args) {

	//FILE *fp;
	//VID vid;

	//if((fp=fopen("/home/abhineet/vid.bin", "wb+"))==NULL) {
	//	printf("Cannot open file.\n");
	//	return NULL;
	//}
	//fread(&vid, sizeof(VID), 1, fp);
	//fprintf(stdout, "Starting getFrame\n");

    PyArrayObject *in_img_py;	
	int current_frame=0;

    /*parse first input array*/
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &in_img_py))
        return NULL;
 //   if (in_img_py == NULL){
	//	fprintf(stdout, "\n----getFrame: in_img_py is NULL----\n\n");
 //       return NULL;
	//}
	//fprintf(stdout, "vid=%d\n", vid);
	//fprintf(stdout, "in_img_py->nd=%d\n", (int)in_img_py->nd);
	//fprintf(stdout, "strides=%d, %d, %d\n",
	//	(int)in_img_py->strides[0], (int)in_img_py->strides[1], (int)in_img_py->strides[2]);
	//fprintf(stdout, "img_width=%d img_height=%d, current_frame=%d\n",
	//	img_width, img_height, current_frame);
    //if (in_img_py->nd != 3) {		
    //    fprintf(stdout, "\n----getFrame: Expected a 3D array----\n\n");
    //    return NULL;
    //}
    //int nchannels=in_img_py->dimensions[2];
    //if (nchannels!=NCHANNELS){
    //	printf("getFrame: Incorrect channel count: Expected: %d Found: %d\n",
    //		NCHANNELS, nchannels);
    //	return NULL;
    //}
    //if ((in_img_py->dimensions[0] != img_height) ||
    //        (in_img_py->dimensions[1] != img_width) ||
    //        (in_img_py->dimensions[2] != NCHANNELS)) {
    //    fprintf(stdout, "\n----getFrame: Incorrect array dimensions: Expected: (%d, %d, %d) Found: (%d, %d, %d)----\\n",
    //           img_height, img_width, NCHANNELS,
    //           (int)in_img_py->dimensions[0], (int)in_img_py->dimensions[1], (int)in_img_py->dimensions[2]);
    //    return NULL;
    //}

    vid->initiate_acquire((current_frame + 1) % n_buffers);
    vid->wait_for_completion(current_frame);


    XVImageRGB<IMG_FMT> & xv_frame=vid->frame(current_frame);
    char * pix=(char*)(xv_frame.data());

	for(int row=0; row<img_height; row++) {
		for(int col=0; col<img_width; col++) {
			int np_location=col*in_img_py->strides[1]+row*in_img_py->strides[0];
			int xv_location=col*xv_nch+row*xv_row_size;
			for(int ch=0; ch<NCHANNELS; ch++) {
				*(in_img_py->data+np_location+(ch*in_img_py->strides[2])) = pix[xv_location+ch];
			}
		}		
	}
	current_frame = (current_frame + 1) % n_buffers;

	//int xv_location=0, np_location=0;
	//for(int row=0; row<img_height; row++) {
	//	for(int col=0; col<img_width; col++) {
	//		//int np_location=col*img->strides[1]+row*img->strides[0];
	//		//int xv_location=col*xv_nch+row*xv_row_size;
	//		for(int ch=0; ch<NCHANNELS; ch++) {
	//			in_img_py->data[np_location] = pix[xv_location];
	//			np_location+=in_img_py->strides[2];
	//			xv_location++;
	//		}
	//		xv_location++;
	//	}
	//}
	//fwrite(&vid, sizeof(VID), 1, fp);
	//fclose(fp);
    return Py_BuildValue("i", 1);
}

static PyObject* getFrame2(PyObject* self, PyObject* args) {

	//printf("getFrame2: img=%d\n", img);

	vid->initiate_acquire((current_frame + 1) % n_buffers);
	//printf("getFrame2: Done initiate_acquire\n");
	vid->wait_for_completion(current_frame);
	//printf("getFrame2: Done wait_for_completion\n");


	XVImageRGB<IMG_FMT> & xv_frame=vid->frame(current_frame);
	char * pix=(char*)(xv_frame.data());

	
	//for(int row=0; row<img_height; row++) {
	//	//printf("getFrame2: row=%d\n", row);
	//    for(int col=0; col<img_width; col++) {
	//        int np_location=col*img->strides[1]+row*img->strides[0];
	//        int xv_location=col*xv_nch+row*xv_row_size;
	//        for(int ch=0; ch<NCHANNELS; ch++) {
	//            *(img->data+np_location+(ch*img->strides[2])) = pix[xv_location+ch];
	//        }
	//    }
	//}

	int xv_location=0, np_location=0;
	for(int row=0; row<img_height; row++) {
		for(int col=0; col<img_width; col++) {			
			for(int ch=0; ch<NCHANNELS; ch++) {
				img->data[np_location] = pix[xv_location];
				np_location+=img->strides[2];
				xv_location++;
			}
			xv_location++;
		}
	}
	current_frame = (current_frame + 1) % n_buffers;
	//printf("getFrame2: Done assigning pixels\n");
	return Py_BuildValue("O", img);
}

static PyObject* close(PyObject* self, PyObject* args) {
	//vid->close();
	return Py_BuildValue("i", 1);
}
