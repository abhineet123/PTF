import numpy
import math
from numpy import matrix as MA

def images_angle(I1,I2):
	dt = numpy.dtype('f8');
	I1v = I1.flatten(1)
	col = MA(I1v.shape).item(1);
	I2v = I2.flatten(1)
	I1vn = I1v/(numpy.sqrt(numpy.sum(numpy.multiply(I1v,I1v))) + 1e-14);
	I2vn = I2v/(numpy.sqrt(numpy.sum(numpy.multiply(I2v,I2v)))+ 1e-14);
	angle = math.degrees(math.acos(numpy.sum(numpy.multiply(I1vn,I2vn))))
	return angle
