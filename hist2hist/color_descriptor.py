import sys
import cv2
import numpy as np
from PIL import Image
from functools import reduce
from histogram import SelfHist


class HistDescriptor:
    def __init__(self, bins, channel_type='HSV'):
        # Store the number of bin for the histogram
        self.channel_type = channel_type
        self.bins = bins
        self.length = reduce(lambda x,y:x*y,[len(i)-1 if isinstance(i, np.ndarray) else i for i in self.bins ])
        
    def describe(self, path):
        hist = np.zeros(self.length,)
        #print(hist.shape)
        try:

            image = self._readimg(path)
            # resize to speed up
            image = cv2.resize(image, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_NEAREST)  
            # Convert image to HSV color
            channel = image.shape[2]
            mask = None
            if channel == 4:
                mask = image[:, :, 3]
            
            if self.channel_type == 'HSV':
                image = cv2.cvtColor(image[:,:,:3], cv2.COLOR_BGR2HSV)
                hist = self._histogram(image, mask, max_value=[180, 255, 255])
            else:
                image = cv2.cvtColor(image[:,:,:3], cv2.COLOR_BGR2RGB)
                hist = self._histogram(image, mask, max_value=[255, 255, 255])
            

        except Exception as e:
            print('url:{}, error:{}'.format(path, e))
            pass
        
        return hist

    def _histogram(self, image, mask, max_value):
        #hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 181, 0, 256, 0, 256])
        hist = SelfHist(image, mask, Bins=self.bins, max_value=max_value, channel_type=self.channel_type)

        return hist
    
    def _readimg(self, path):
        if isinstance(path, str):
            if path.endswith('gif'):
                img = np.array(Image.open(path).convert('RGB'))
                img = img[:,:,::-1]
            else:
                img = cv2.imread(path, -1)
        else:
            img = cv2.cvtColor(path, cv2.COLOR_RGBA2BGRA)
            
        return img
            



if __name__ == '__main__':
    path = sys.argv[1]

    H_Bin = np.array([0, 21, 41, 76, 156, 191, 271, 296, 316, 361]) / 2.0
    hsvdescriptor = HistDescriptor([H_Bin, 8, 8], channel_type='HSV')
    rgbdescriptor = HistDescriptor([8, 8, 8], channel_type='RGB')


    print(rgbdescriptor.describe(path))