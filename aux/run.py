# -*- coding: utf-8 -*-
"""
Trains MADE on Binarized MNIST, which can be downloaded here:
https://github.com/mgermain/MADE/releases/download/ICML2015/binarized_mnist.npz
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab
from made import MADE
import torch.distributions.binomial
import os
from torchvision.utils import save_image
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Sample and Run_Epoch functions: --------------------------------------------------------------------
def run_epoch(split, upto=None):
    torch.set_grad_enabled(split=='train') # enable/disable grad for efficiency of forwarding test batches
    model.train() if split == 'train' else model.eval()
    nsamples = 1 if split == 'train' else args.samples
    x = xtr if split == 'train' else xte
    N,D = x.size() # N is the number of samples and D is the size of each sample
    			   # In our case 60.000x784 or 10.000x784 are the sizes.
    B = 100 # batch size, less than in the loaded code!
    nsteps = N//B if upto is None else min(N//B, upto) # enough steps so that we use the whole set
    lossfs = []
    for step in range(nsteps):

        # fetch the next batch of data
        xb = Variable(x[step*B:step*B+B])
        # xb = x[step*B:step*B+B]
        xb = xb.float()

        # print(xb.dtype)

        # get the logits, potentially run the same batch a number of times, resampling each time
        xbhat = torch.zeros_like(xb)
        for s in range(nsamples):
            # perform order/connectivity-agnostic training by resampling the masks
            if step % args.resample_every == 0 or split == 'test': # if in test, cycle masks every time
                model.update_masks()
            # forward the model
            xbhat += model(xb)
        xbhat /= nsamples

        # evaluate the binary cross entropy loss
        loss = F.binary_cross_entropy(xbhat, xb, size_average=False) / B # With logits before...
        lossf = loss.data.item()
        lossfs.append(lossf)

        # backward/update
        if split == 'train':
            opt.zero_grad()
            loss.backward()            
            opt.step()

    print("%s epoch average loss: %f" % (split, np.mean(lossfs)))

    if split == 'train':
        trainL[epoch-1] = np.mean(lossfs)

    if split == 'test':
        testL[epoch-1] = np.mean(lossfs)

def sample(n):
    image = np.zeros(28*28)
    image = torch.tensor(image)
    for i in range(28*28):
        # for idx, m in enumerate(model.named_modules()):
        #     print(idx, '->', m)
        prob = model(image) # Why not outputting between 0 and 1?¿?
        # print(prob[0:20])
        pixel = np.random.binomial(1, prob[i].detach().numpy())
        image[i] = pixel
    # Now image stores the sampled image using the regular order...
    path = 'samplesS/sample_' + str(n) + '.png'
    plt.imsave(path, np.array(image).reshape(28,28), cmap=cm.gray)
    return np.array(image).reshape(1, 28, 28)
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-d', '--data-path', required=True, type=str, help="Path to binarized_mnist.npz")
    parser.add_argument('-dtr', '--dtr-path', required=True, type=str, help="Path to x_train.npy")
    parser.add_argument('-dte', '--dte-path', required=True, type=str, help="Path to x_test.npy")
    parser.add_argument('-q', '--hiddens', type=str, default='500', help="Comma separated sizes for hidden layers, e.g. 500, or 500,500")
    parser.add_argument('-n', '--num-masks', type=int, default=1, help="Number of orderings for order/connection-agnostic training")
    parser.add_argument('-r', '--resample-every', type=int, default=20, help="For efficiency we can choose to resample orders/masks only once every this many steps")
    parser.add_argument('-s', '--samples', type=int, default=1, help="How many samples of connectivity/masks to average logits over during inference")
    args = parser.parse_args()
    # --------------------------------------------------------------------------

    # Hyper-parameters:
    np.random.seed(42)
    torch.manual_seed(42)
    num_epochs = 0 # 150
    NN = True
    # torch.cuda.manual_seed_all(42)

    # IMAGE PREPROCESSING:

    # print("loading binarized mnist") #, args.data_path)
    # mnist = np.load(args.data_path)
    xtr = np.load(args.dtr_path)
    xte = np.load(args.dte_path)
    # xtr, xte = mnist['train_data'], mnist['valid_data']
	# xtr = torch.from_numpy(xtr).cuda()
    # xte = torch.from_numpy(xte).cuda()
    xtr = torch.from_numpy(xtr)
    xte = torch.from_numpy(xte)

    # construct model and ship to GPU

    # Recall, map(fun,iter) applies the function to every element of the iter.
    hidden_list = list(map(int, args.hiddens.split(',')))

    print("The size of the training set is ", xtr.size())
    print("The size of the test set is ", xte.size())
    # 60000x784, 10000x784, recall that 28*28 = 784

    # MODEL and optimizer:
    model = MADE(xtr.size(1), hidden_list, xtr.size(1), num_masks=args.num_masks)
    print("number of model parameters:",sum([np.prod(p.size()) for p in model.parameters()]))
    # model.cuda()
    model = model.float()

    # set up the optimizer
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4) # Initially -3
    # Here we apply weight decay to the learning rate, every 45 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=45, gamma=0.1)

    
    # TRAINING:
    
    state_dict = torch.load('save/model_final.ckpt')
    model.load_state_dict(state_dict)

    testL = np.zeros(num_epochs)
    trainL = np.zeros(num_epochs)

    for epoch in range(1,num_epochs+1):
        print("epoch %d" % (epoch, ))
        scheduler.step(epoch)
        run_epoch('test', upto=5) # run only a few batches for approximate test accuracy
        run_epoch('train')
        sample(epoch)
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join('save', 'model--{}.ckpt'.format(epoch+1)))

        plt.figure()
        pylab.xlim(0, num_epochs + 1)
        plt.plot(range(1, num_epochs + 1), trainL, label='train loss')
        plt.plot(range(1, num_epochs + 1), testL, label='test loss')
        plt.legend()
        plt.savefig(os.path.join('save', 'loss.pdf'))
        plt.close()

    torch.save(model.state_dict(), os.path.join('save', 'model_final.ckpt'))
    print("optimization done. full test set eval:") # 79.72 my last experiment with 100 epochs!
    if num_epochs > 0:
        run_epoch('test')

    ############


    # NEAREST NEIGHBOUR!!!!!
    # Images 28x28, search the closest one.
    # function(generated_image) --> closest training_image

    if NN == True:
        aux_data_loader = np.load(args.dtr_path)

        def nearest_gt(generated_image):
            min_d = 0
            closest = False
            for i, image in enumerate(aux_data_loader):
                image = np.array(image).reshape(28,28) # all distances in binary...
                image = torch.tensor(image).float()
                d = torch.dist(generated_image,image) # must be torch tensors (1,28,28)
                if i == 0 or d < min_d:
                    min_d = d
                    closest = image

            return closest

        # calculate closest to...
        samples = torch.zeros(24, 1, 28, 28)
        NN = torch.zeros(24, 1, 28, 28)
        for i in range(0,24):
                image = torch.tensor(sample(i))
                samples[i] = image
                NN[i] = nearest_gt(samples[i])
                print(i)
        save_image(samples, 'f24.png')
        save_image(NN.data, 'NN24.png')





