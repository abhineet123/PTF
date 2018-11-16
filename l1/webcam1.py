import cv2;
if __name__ == "__main__":
     camera =  cv2.VideoCapture(0);
     while True:
          f,img = camera.read();
          cv2.imshow("webcam",img);
          if (cv2.waitKey (5) != -1):
                break;