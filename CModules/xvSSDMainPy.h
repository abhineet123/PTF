#include <XVSSD.h>
#include <XVTracker.h>
#include <XVMpeg.h>
#include <XVDig1394.h>
#include <XVV4L2.h>
#include <XVWindowX.h>
#define PIX_TYPE XV_RGB24

typedef XVImageRGB< PIX_TYPE > IMAGE_TYPE;
typedef XVVideo< IMAGE_TYPE > VID;
typedef XVInteractWindowX< PIX_TYPE > WIN_INT;

class XVSSDMainPy {

private:
	char* xv_data;
	IMAGE_TYPE* xv_frame;	
	XVSize *xv_size;

	int current_frame;
	int show_xv_window;
	int steps_per_frame;
	int direct_capture;
	int n_buffers;

public:
	virtual void initialize(Mat &in_img, double pos_x, double pos_y,
		double size_x, double size_y, bool direct_capture, bool show_xv_window);
	virtual void initialize(Mat &in_img, double pos_x, double pos_y,
		double size_x, double size_y, double no_of_levels, double scale, 
		bool direct_capture, bool show_xv_window);

    /* ==== update tracker ==== */
    void update(Mat &in_img_py) {
        current_tracker=trackers[current_tracker_id];
        if (current_tracker->direct_capture) {
            xv_frame=&(vid->frame(current_tracker->current_frame));
            current_tracker->current_frame = (current_tracker->current_frame + 1) % n_buffers;
        } else {
            numpyToXV();
        }
        for(int i = 0; i < current_tracker->steps_per_frame-1; ++i) {
            current_tracker->ssd->step(*xv_frame);
        }

        STATE_PAIR_TYPE current_state=current_tracker->ssd->step(*xv_frame);

        updateCorners(current_state);

        for(int i=0; i<NCORNERS; i++) {
            corners_data[i]=corners[i].PosX();
            corners_data[i+corners_array->strides[1]]=corners[i].PosY();
        }
        if(current_tracker->show_xv_window) {
            if(current_tracker_id==first_win_id) {
                win_int->CopySubImage(*xv_frame);
            }
            win_int->drawLine(corners[0].PosX(), corners[0].PosY(), corners[1].PosX(), corners[1].PosY(), DEFAULT_COLOR, xv_line_width);
            win_int->drawLine(corners[1].PosX(), corners[1].PosY(), corners[2].PosX(), corners[2].PosY(), DEFAULT_COLOR, xv_line_width);
            win_int->drawLine(corners[2].PosX(), corners[2].PosY(), corners[3].PosX(), corners[3].PosY(), DEFAULT_COLOR, xv_line_width);
            win_int->drawLine(corners[3].PosX(), corners[3].PosY(), corners[0].PosX(), corners[0].PosY(), DEFAULT_COLOR, xv_line_width);
            if(current_tracker_id==last_win_id) {
                win_int->swap_buffers();
                win_int->flush();
            }
        }

        current_tracker_id=(current_tracker_id+1) % no_of_trackers;
        return Py_BuildValue("O", corners_array);
    }

	inline void numpyToXV() {
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
				xv_location++;
			}
			//printf("\n------------------------------\n");
		}
	}
};