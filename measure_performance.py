from CNN.utils import *

from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import pickle

parser = argparse.ArgumentParser(description='Predict the network accuracy.')
parser.add_argument('parameters', metavar = 'parameters', help='name of file parameters were saved in. These parameters will be used to measure the accuracy.')

if __name__ == '__main__':
    args = parser.parse_args()
    save_path = args.parameters
    
    params, cost = pickle.load(open(save_path, 'rb'))
    [f1, f2, w3, w4, b1, b2, b3, b4] = params
    
    # Get test data
    m = 144
    X = extract_data('dataset_50x50_MNIST/t10k-images-idx3-ubyte.gz', m, 50)
    y_dash = extract_labels('dataset_50x50_MNIST/t10k-labels-idx1-ubyte.gz', m).reshape(m,1)

    image_number = 3
    image = X[image_number].reshape(50, 50, 3)
    image -= int(np.mean(image))
    image /= int(np.std(image))
    print(y_dash[image_number])
    plt.imshow(image)
    plt.show()
    
    # Normalize the data
    X-= int(np.mean(X)) # subtract mean
    X/= int(np.std(X)) # divide by standard deviation
    test_data = np.hstack((X,y_dash))
    
    X = test_data[:,0:-1]
    X = X.reshape(len(test_data), 3, 50, 50)
    y = test_data[:,-1]

    corr = 0
    digit_count = [0 for i in range(2)]
    digit_correct = [0 for i in range(2)]
   
    print()
    print("Computing accuracy over test set:")

    t = tqdm(range(len(X)), leave=True)
    print(y)
    for i in t:
        x = X[i]
        pred, prob = predict(x, f1, f2, w3, w4, b1, b2, b3, b4)
        ##digit_count[int(y[i])]+=1
        if pred==y[i]:
            corr+=1
            ##digit_correct[pred]+=1
        acc = 0
        if(float(corr/(i+1))*100 < 50):
            acc = 100 - float(corr/(i+1))*100
        else:
            acc = float(corr/(i+1))*100
        t.set_description("Acc:%0.2f%%" % acc)

    acc = 0
    if(float(corr/(i+1))*100 < 50):
        acc = 100 - float(corr/(i+1))*100
    else:
        acc = float(corr/(i+1))*100
        
    print("Overall Accuracy: %.2f" % acc)
    x = np.arange(10)
    #digit_recall = [x/y for x,y in zip(digit_correct, digit_count)]
    plt.xlabel('Digits')
    plt.ylabel('Recall')
    plt.title("Recall on Test Set")
    #plt.bar(x,digit_recall)
    plt.show()