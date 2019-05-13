import pytorch_model as model
from glob import glob
import numpy as np
from PIL import Image
import time
paths = glob('./test/*.*')

if __name__ =='__main__':
    for path in paths:
        im = Image.open(path)
        img = np.array(im.convert('RGB'))
        t = time.time()
        result, img, angle = model.model(img, detectAngle=True)
        print("---------------------------------------")
        print("It takes time:{}s".format(time.time()-t))
        print("File Path: ".format(path),
              "Image text angle is: {} Degree".format(angle),
              "Result:\n")

        for key in result:
            print(result[key][1])
