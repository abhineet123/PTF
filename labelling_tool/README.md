LabelImg
========

LabelImg is a graphical image annotation tool.

It is written in Python and uses Qt for its graphical interface.

Annotations are saved as XML files in PASCAL VOC format, the format used
by [ImageNet](http://www.image-net.org/).

Watch demo videos:

\<<https://youtu.be/x2CK0uEK4tM>\>

\<<https://youtu.be/5iNgnzek-0U>\>

\<<https://youtu.be/ajYzLXlUJJM>\>

\<<https://youtu.be/mxd20Oro32U>\>

\<<https://youtu.be/rzQ7qpaSPVA>\>

\<<https://youtu.be/Tn-IrpM9J6Q>\>

\<<https://youtu.be/PiQCrTWaidc>\>

\<<https://youtu.be/BMulaUS3Z6c>\>

\<<https://youtu.be/s_YdPMhsvE4>\>

\<<https://youtu.be/FsIO4oVWKZo>\>

\<<https://youtu.be/QedayopySCg>\>

Installation
------------

Linux/Ubuntu/Mac requires at least [Python
2.6](http://www.python.org/getit/) and has been tested with [PyQt
4.8](http://www.riverbankcomputing.co.uk/software/pyqt/intro).

### Ubuntu Linux

Python 2 + Qt4

``` {.sourceCode .}
sudo apt-get install pyqt4-dev-tools
sudo pip install lxml
make qt4py2
python labelImg.py
python labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```

Python 3 + Qt5

``` {.sourceCode .}
sudo apt-get install pyqt5-dev-tools
sudo pip3 install lxml
make qt5py3
python3 labelImg.py
python3 labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```

### OS X

Python 2 + Qt4

``` {.sourceCode .}
brew install qt qt4
brew install libxml2
make qt4py2
python labelImg.py
python  labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```

### Windows

Download and setup [Python 2.6 or
later](https://www.python.org/downloads/windows/),
[PyQt4](https://www.riverbankcomputing.com/software/pyqt/download) and
[install lxml](http://lxml.de/installation.html).

Open cmd and go to [labelImg](#labelimg) directory

``` {.sourceCode .}
pyrcc4 -o resources.py resources.qrc
python labelImg.py
python labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```

Usage
-----

#### Steps

1.  Build and launch using the instructions above.
2.  Click 'Change default saved annotation folder' in Menu/File
3.  Click 'Open Dir'
4.  Click 'Create RectBox'
5.  Click and release left mouse to select a region to annotate the rect
    box
6.  You can use right mouse to drag the rect box to copy or move it

The annotation will be saved to the folder you specify.

