
import numpy as np
from scipy.io import loadmat
from PIL import Image
import os
from pathlib import Path

def main():

    directory = '/home/robotics-verse/projects/felix/DataSet/nmi-vasc-robot/data/dus_test'
    for subdir in os.listdir(directory):
        for file in os.listdir(directory + '/' + subdir):
            mat = loadmat(directory + '/' + subdir + '/' + file)
            
            for colname in mat.keys():

                # ignore private column names
                if colname.startswith("__") or colname == 'Label_info':
                    continue

                element = mat[colname]

                if isinstance(element, np.ndarray):
                    im = Image.fromarray((element * 255).astype(np.uint8))
                    im = im.convert('RGB') if colname =='Image_doppler' else im.convert('L')
                    im.save('{}/{}/{}_{}.png'.format(directory, subdir, colname, file.split('.')[0]))


if __name__ == "__main__":

    main()