#include <windows.h>
#include <Python.h>
#include <numpy/arrayobject.h>

using namespace std;

static PyObject* hideBorder(PyObject* self, PyObject* args);
static PyObject* hideBorder2(PyObject* self, PyObject* args);
static PyObject* loseFocus(PyObject* self, PyObject* args);
static PyObject* showWindow(PyObject* self, PyObject* args);
static PyObject* hideWindow(PyObject* self, PyObject* args);

static PyMethodDef winUtilsMethods[] = {
	{ "hideBorder", hideBorder, METH_VARARGS },
	{ "hideBorder2", hideBorder2, METH_VARARGS },
	{ "loseFocus", loseFocus, METH_VARARGS },
	{ "showWindow", showWindow, METH_VARARGS },
	{ "hideWindow", hideWindow, METH_VARARGS },
	{ NULL, NULL, 0, NULL }     /* Sentinel - marks the end of this structure */
};

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initwinUtils() {
	(void)Py_InitModule("winUtils", winUtilsMethods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}
#else
static struct PyModuleDef winUtilsModule = {
	PyModuleDef_HEAD_INIT,
	"winUtils",   /* name of module */
	NULL, /* module documentation, may be NULL */
	-1,       /* size of per-interpreter state of the module,
			  or -1 if the module keeps state in global variables. */
	winUtilsMethods
};
PyMODINIT_FUNC PyInit_winUtils(void) {
	import_array();
	return PyModule_Create(&winUtilsModule);
}
#endif

static PyObject* showWindow(PyObject* self, PyObject* args) {
	char* win_name;
	int x, y, w, h;
	if (!PyArg_ParseTuple(args, "z", &win_name)) {
		PySys_WriteStdout("\n----winUtils::showWindow: Input arguments could not be parsed----\n\n");
		return Py_BuildValue("i", 0);
	}
	HWND win_handle = FindWindow(0, win_name);
	if (!win_handle) {
		PySys_WriteStdout("winUtils::showWindow: Failed FindWindow\n");
		return Py_BuildValue("i", 0);
	}
	PySys_WriteStdout("winUtils::showWindow\n");
	ShowWindow(win_handle, SW_SHOW);
}


static PyObject* hideWindow(PyObject* self, PyObject* args) {
	char* win_name;
	int x, y, w, h;
	if (!PyArg_ParseTuple(args, "z", &win_name)) {
		PySys_WriteStdout("\n----winUtils::hideWindow: Input arguments could not be parsed----\n\n");
		return Py_BuildValue("i", 0);
	}
	HWND win_handle = FindWindow(0, win_name);
	if (!win_handle) {
		PySys_WriteStdout("winUtils::hideWindow: Failed FindWindow\n");
		return Py_BuildValue("i", 0);
	}
	PySys_WriteStdout("winUtils::hideWindow\n");
	ShowWindow(win_handle, SW_HIDE);
}


static PyObject* hideBorder(PyObject* self, PyObject* args) {
	char* win_name;
	int x, y, w, h;
	if(!PyArg_ParseTuple(args, "iiiiz", &x, &y, &w, &h, &win_name)) {
		PySys_WriteStdout("\n----winUtils::create: Input arguments could not be parsed----\n\n");
		return Py_BuildValue("i", 0);
	}
	HWND win_handle = FindWindow(0, win_name);
	if(!win_handle) {
		PySys_WriteStdout("Failed FindWindow\n");
		return Py_BuildValue("i", 0);
	}

	printf("x: %d\n", x);
	printf("y: %d\n", y);
	printf("w: %d\n", w);
	printf("h: %d\n", h);

	// Resize
	unsigned int flags = (SWP_SHOWWINDOW | SWP_NOSIZE | SWP_NOZORDER);
	flags &= ~SWP_NOSIZE;
	SetWindowPos(win_handle, HWND_TOPMOST, x, y, w, h, flags);

	// Borderless
	SetWindowLong(win_handle, GWL_STYLE, GetWindowLong(win_handle, GWL_EXSTYLE) | WS_EX_TOPMOST);

	return Py_BuildValue("i", 1);
}

static PyObject* hideBorder2(PyObject* self, PyObject* args) {
	char* win_name;
	if(!PyArg_ParseTuple(args, "z", &win_name)) {
		PySys_WriteStdout("\n----winUtils::hideBorder2: Input arguments could not be parsed----\n\n");
		return Py_BuildValue("i", 0);
	}
	HWND win_handle = FindWindow(0, win_name);
	if(!win_handle) {
		PySys_WriteStdout("Failed FindWindow\n");
		return Py_BuildValue("i", 0);
	}
	// change style of the child HighGui window
	DWORD style = ::GetWindowLong(win_handle, GWL_STYLE);
	style &= ~WS_OVERLAPPEDWINDOW;
	style |= WS_POPUP;
	style |= WS_EX_TOPMOST;
	::SetWindowLong(win_handle, GWL_STYLE, style);

	// change style of the parent HighGui window
	HWND hParent = ::FindWindow(0, win_name);
	style = ::GetWindowLong(hParent, GWL_STYLE);
	style &= ~WS_OVERLAPPEDWINDOW;
	style |= WS_POPUP;
	style |= WS_EX_TOPMOST;
	::SetWindowLong(hParent, GWL_STYLE, style);
	SetWindowPos(hParent, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
	return Py_BuildValue("i", 1);
}

static PyObject* loseFocus(PyObject* self, PyObject* args) {
	char* win_name;
	if(!PyArg_ParseTuple(args, "z", &win_name)) {
		PySys_WriteStdout("\n----winUtils::loseFocus: Input arguments could not be parsed----\n\n");
		return Py_BuildValue("i", 0);
	}
	HWND win_handle = FindWindow(0, win_name);
	if(!win_handle) {
		PySys_WriteStdout("Failed FindWindow\n");
		return Py_BuildValue("i", 0);
	}
	SetWindowPos(win_handle, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);

	return Py_BuildValue("i", 1);
}
