from CNN.utils import *

from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from numpy import asarray

parser = argparse.ArgumentParser(description='Predict the network accuracy.')
parser.add_argument('parameters', metavar = 'parameters', help='name of file parameters were saved in. These parameters will be used to measure the accuracy.')

if __name__ == '__main__':
    args = parser.parse_args()
    save_path = args.parameters
    
    params, cost = pickle.load(open(save_path, 'rb'))
    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    image = Image.open('test_images/plastic3.jpg')
    new_image = image.resize((50, 50))
    X = asarray(new_image)
    img = Image.fromarray(X)
    img.show()

    # Normalize the data
    X_= X - int(np.mean(X)) # subtract mean
    X_ = X_ /int(np.std(X_)) # divide by standard deviation
    
    X_ = X_.reshape(3, 50, 50)

    pred, prob = predict(X_, f1, f2, w3, w4, b1, b2, b3, b4)

    if(pred == 1): 
        print("Plastic")
    else:
        print("non-plastic")