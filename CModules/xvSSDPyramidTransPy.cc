#include "xvSSDPyramidTransPy.h"

/* ==== Set up the methods table ====================== */
static PyMethodDef xvInputMethods[] = {
	{"initialize", initialize, METH_VARARGS},
	{"update", update, METH_VARARGS},
	{NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the C_test functions ====================== */
// Module name must be xvInput in compile and linked
PyMODINIT_FUNC initxvSSDPyramidTransPy()  {
	(void) Py_InitModule("xvSSDPyramidTransPy", xvInputMethods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

/* ==== initialize tracker ==== */

static PyObject* initialize(PyObject* self, PyObject* args) {

	//fprintf(stdout, "Starting initialize\n");
	double pos_x, pos_y, size_x, size_y;
	TrackerStruct *new_tracker=new TrackerStruct;

	/*parse first input array*/
	if (!PyArg_ParseTuple(args, "O!ddddiidii", &PyArray_Type, &in_img_py, &pos_x, &pos_y,
		&size_x, &size_y, &new_tracker->steps_per_frame,
		&new_tracker->no_of_levels, &new_tracker->scale,
		&new_tracker->direct_capture, &new_tracker->show_xv_window)) {
			printf("\n----xvSSDPyramidalTrans::initialize: Input arguments could not be parsed----\n\n");
			return NULL;
	}

	if (in_img_py == NULL) {
		printf("\n----xvSSDPyramidalTrans::update: in_img_py is NULL----\n\n");
		return NULL;
	}

	printf("Initializing Pyramidal Trans SSD Tracker with:\n\t");
	printf("size_x=%f\n\t",size_x);
	printf("size_y=%f\n\t",size_y);
	printf("no_of_levels=%d\n\t",new_tracker->no_of_levels);
	printf("scale=%f\n\t",new_tracker->scale);
	printf("steps_per_frame=%d\n",new_tracker->steps_per_frame);

	if (new_tracker->direct_capture) {
		printf("Direct capture is enabled\n");
		if(vid) {
			printf("XVision video stream has already been initialized\n");
		} else {
			FILE *fp=fopen("xvInput.bin", "rb");
			if (fp==NULL) {
				fprintf(stdout, "xvSSDPyramidalTrans::initialize: File xvInput.bin could not be opened for reading\n");
				return NULL;
			}
			fread(&vid, sizeof(*vid), 1, fp);
			fclose(fp);
			n_buffers = vid->buffer_count();
		}
		if(!vid) {
			printf("xvSSDPyramidalTrans::initialize: vid was not initialized successfully\n");
			return NULL;
		} else {
			xv_frame=&(vid->frame(new_tracker->current_frame));
			new_tracker->current_frame = (new_tracker->current_frame + 1) % n_buffers;
		}
	} else if (no_of_trackers==0) {
		img_height=in_img_py->dimensions[0];
		img_width=in_img_py->dimensions[1];

		printf("xvSSDPyramidalTrans::initialize: img_width=%d img_height=%d\n", img_width, img_height);

		xv_frame = new IMAGE_TYPE(img_width, img_height);
		xv_data = (char*) (xv_frame->data());

		numpyToXV();
	}

	int dims[]= {2, NCORNERS};
	corners_array=(PyArrayObject *)PyArray_FromDims(2,dims,NPY_INT32);
	corners_data=(int*)corners_array->data;

	double min_x = pos_x - size_x / 2.0;
	double min_y = pos_y - size_y / 2.0;
	XVPosition init_pos(pos_x, pos_y);
	XVPosition init_pos_min(min_x, min_y);
	new_tracker->xv_size=new XVSize(size_x, size_y);
	XVROI roi( *(new_tracker->xv_size), init_pos_min );
	IMAGE_TYPE init_template = subimage( *xv_frame, roi );

	new_tracker->stepper=new STEPPER_TYPE(init_template, new_tracker->scale, new_tracker->no_of_levels);
	new_tracker->ssd=new TRACKER_TYPE;
	new_tracker->ssd->setStepper(*(new_tracker->stepper));
	new_tracker->ssd->initState((STATE_TYPE)(init_pos));

	if(new_tracker->show_xv_window) {
		last_win_id=no_of_trackers;
		if (!win_int) {
			first_win_id=no_of_trackers;
			printf("Initializing xv window\n");
			win_int=new WIN_INT(*xv_frame);
			printf("mapping xv window\n");
			win_int->map();
		}
	}
	printf("Pushing back new tracker\n");
	trackers.push_back(new_tracker);
	no_of_trackers++;

	return Py_BuildValue("i", 1);
}

/* ==== update tracker ==== */
static PyObject* update(PyObject* self, PyObject* args) {

	current_tracker=trackers[current_tracker_id];

	if (current_tracker->direct_capture) {
		//printf("\n----xvSSDPyramidalTrans: update: Using direct capture----\n\n");
		xv_frame=&(vid->frame(current_tracker->current_frame));
		current_tracker->current_frame = (current_tracker->current_frame + 1) % n_buffers;
	} else {
		/*parse first input array*/
		if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &in_img_py)) {
			printf("\n----xvSSDPyramidalTrans::update: Input arguments could not be parsed----\n\n");
			return NULL;
		}

		if (in_img_py == NULL) {
			printf("\n----xvSSDPyramidalTrans::update: in_img_py is NULL----\n\n");
			return NULL;
		}
		numpyToXV();
	}

	for(int i = 0; i < current_tracker->steps_per_frame-1; ++i) {
		current_tracker->ssd->step(*xv_frame);
	}

	STATE_PAIR_TYPE current_state=current_tracker->ssd->step(*xv_frame);

	/*double posx=current_state.state.PosX();
	double posy=current_state.state.PosY();*/

	updateCorners(current_state);

	//printf("Current corners:\n");

	for(int i=0; i<NCORNERS; i++) {
		//printf("%d\t %d\n", corners[i].PosX(), corners[i].PosY());
		corners_data[i]=corners[i].PosX();
		corners_data[i+corners_array->strides[1]]=corners[i].PosY();

		/*int j=(i+1) % NCORNERS;
		win_int->drawLine(corners[i].PosX(), corners[i].PosY(),
		corners[j].PosX(), corners[j].PosY());*/

	}
	//printf("-------------------------\n");
	//ssd->show(*win_int);
	if(current_tracker->show_xv_window) {
		if(current_tracker_id==first_win_id) {
			win_int->CopySubImage(*xv_frame);
		}
		win_int->drawLine(corners[0].PosX(), corners[0].PosY(), corners[1].PosX(), corners[1].PosY(), DEFAULT_COLOR, xv_line_width);
		win_int->drawLine(corners[1].PosX(), corners[1].PosY(), corners[2].PosX(), corners[2].PosY(), DEFAULT_COLOR, xv_line_width);
		win_int->drawLine(corners[2].PosX(), corners[2].PosY(), corners[3].PosX(), corners[3].PosY(), DEFAULT_COLOR, xv_line_width);
		win_int->drawLine(corners[3].PosX(), corners[3].PosY(), corners[0].PosX(), corners[0].PosY(), DEFAULT_COLOR, xv_line_width);
		//ssd->show(*win_int);
		if(current_tracker_id==last_win_id) {
			win_int->swap_buffers();
			win_int->flush();
		}
	}

	current_tracker_id=(current_tracker_id+1) % no_of_trackers;

	return Py_BuildValue("O", corners_array);
}
