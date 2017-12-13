import cv2
import numpy as np
import os

class ImgCapture:
    DIR_NAME = 'data'


    def captureData(self):
        cap = cv2.VideoCapture(0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('video.avi',fourcc, 20.0, (w,h))

        while(True):
            ret, frame = cap.read()
            try:
                out.write(frame)
            except:
                print('ERROR - Not writting to file') 
            cv2.imshow('Capture DataFrame', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('p'):
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                break


    def generateData(self):
        cap = cv2.VideoCapture('video.avi')

        try:
            if not os.path.exists(self.DIR_NAME):
                os.makedirs(self.DIR_NAME)
        except OSError:
            print ('Error: Creating directory {}'.format(self.DIR_NAME))

        currentFrame = 0

        while(currentFrame < 100): 
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Saves image of the current frame in jpg file
            name = './'+ self.DIR_NAME + '/frame' + str(currentFrame) + '.jpg'
        #     print ('Creating...' + name)
            cv2.imwrite(name, frame)

            # To stop duplicate images
            currentFrame += 1
        print ('Dataset Ready!')
        
        # When everything done, release the capture
        cap.release()
