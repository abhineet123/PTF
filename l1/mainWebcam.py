from L1TrackingBPR_APGupWebcam import L1TrackingBPR_APGupWebcam
from numpy import matrix as MA




	#Initialization for the first frame. 
	#Each column is a point indicating a corner of the target in the first image. 
	#The 1st row is the y coordinate and the 2nd row is for x.
	#Let [p1 p2 p3] be the three points, they are used to determine the affine parameters of the target, as following
	# p1(65,55)-----------p3(170,53)
	#         |                 |
	#         |     target      |
	#         | 		    |
	#   p2(64,140)--------------
	# % size of template    


class para():
	def __init__(self):
		self.lambd = MA([[0.2,0.001,2]])
		self.angle_threshold = 40
		self.Lip = 8 ;
		self.Maxit = 5 ; 
		self.nT = 50; # number of templates for the sparse representation
		self.rel_std_afnv =  MA([[0.003,0.03,0.003,0.03,1,1]]) ; # diviation of the sampling of particle filter
		self.n_sample = 100; # No of particles
		self.sz_T = MA([[12,15]]); # Reshape each image so that they have the same image space representation
		# self.init_pos = MA([[360,520,360], [80,100,150]]); # juice
		# self.init_pos = MA([[280,455,275],[320,330,440]]) # book in shelf
		# self.init_pos = MA([[415,465,415],[360,360,410]])  # drinking from cup
		# self.init_pos = MA([[315,358,315],[302,302,350]]);
		# self.init_pos = MA([[385,500,385],[462,462,600]]); # MARS Rover
#		self.init_pos = MA([[int(pts[0,1]),int(pts[1,1]),int(pts[2,1])],[int(pts[0,0]),int(pts[1,0]),int(pts[2,0])]])
               # self.init_pos = MA([[121,172,121],[58,58,109]]);#SYLV
#		self.bDebug = 0; # debugging indicator
		#self.bShowSaveImag = 1 ; # indicator for result image show and save after tracking finished
def main():
	paraT = para()
	L1TrackingBPR_APGupWebcam(paraT)


if __name__ == '__main__':
	main()
