import os
import torch
from torch import Tensor, nn, optim
from torch.autograd import Variable
from torch.nn import functional as F

from torchvision import datasets

if torch.cuda.is_available():
    print ("CUDA IS AVAILABLE. USING CUDA")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def load_data(small=False, normalize=False):
    data_dir = './cifar'

    train_set = datasets.CIFAR10(data_dir, train=True, download=True)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True)

    train_input = torch.from_numpy(train_set.train_data)
    train_input = train_input.transpose(3, 1).transpose(2, 3).float()
    
    train_target = torch.LongTensor(train_set.train_labels)

    test_input = torch.from_numpy(test_set.test_data).float()
    test_input = test_input.transpose(3, 1).transpose(2, 3).float()
    
    test_target = torch.LongTensor(test_set.test_labels)

    print(train_input.size(0), test_input.size(0))

    if (torch.cuda.is_available()):
        train_input = train_input.cuda()
        train_target = train_target.cuda()
        test_input = test_input.cuda()
        test_target = test_target.cuda()

    if small:
        print ('** Using small set of 1000 samples')
        train_input = train_input.narrow(0, 0, 1000)
        train_target = train_target.narrow(0, 0, 1000)

        test_input = test_input.narrow(0, 0, 1000)
        test_target = test_target.narrow(0, 0, 1000)

    if normalize:
        mean, std = train_input.mean(), train_input.std()
        train_input.sub_(mean).div_(std)
        test_input.sub_(mean).div_(std)

    return train_input, train_target, test_input, test_target



    

train_input, train_target, test_input, test_target = load_data(small=False)

train_input, train_target = Variable(train_input), Variable(train_target)
test_input, test_target = Variable(test_input), Variable(test_target)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        return x



"""
input:  3x32x32

conv1-1:16x32x32
relu
conv1-2:16x32x32
relu
pool1-1:16x16x16

conv2-1:32x16x16
relu
conv2-2:32x16x16
relu
pool2-1:32x8x8

conv3-1:64x8x8
relu
conv3-2:64x8x8
relu
pool3-1:64x4x4

input2:1x1024
fc1:1024->1024
fc2:1024->10
softmax
"""


def create_conv_blocks():
    return [nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2)]

def create_classifier():
    return [Flatten(),
    nn.Linear(1024, 1024),
    nn.ReLU(inplace=True),
    nn.Linear(1024, 10),
    nn.Softmax()]

def create_model():
    m = create_conv_blocks()
    m.extend(create_classifier())
    model = nn.Sequential(*m)
    return model

def train_model(model, train_input, train_target, batch_size, epochs):
    loss_fn = nn.CrossEntropyLoss()
    eta = 1e-2
    optimizer = optim.SGD(model.parameters(), lr=eta)


    for e in range(epochs):
        sum_loss = 0
        nb_errors = 0
        
        for b in range(0, train_input.size(0), batch_size):
            input = train_input.narrow(0, b, batch_size)
            target = train_target.narrow(0, b, batch_size)
            
            output = model(input)

            # Training accuracy
            _, predicted = torch.max(output.data, 1)
            for i in range(0, batch_size):
                if (target.data[i] != predicted[i]):
                    nb_errors += 1
            
            
            loss = loss_fn(output, target)
            
            # sum_loss += loss.data

            model.zero_grad()
            loss.backward()

            optimizer.step()

            # print statistics
            sum_loss += loss.data
            if b % 2000 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (e + 1, b + 1, sum_loss / 2000))
                sum_loss = 0.0
        
        print('[%5d] accuracy: %.3f' % (e + 1, (1-(nb_errors/train_input.size(0)))*100))


def compute_nb_errors(model, input, target, batch_size):
    nb_errors = 0

    for b in range(0, input.size(0), batch_size):
        output = model(input.narrow(0, b, batch_size))
        _, predicted = torch.max(output.data, 1)
        # predicted = int(predicted)

        for i in range(0, batch_size):
            if (target.data[b+i] != predicted[i]):
                nb_errors += 1
    return nb_errors

# Hyper parameters
batch_size = 100

# iteration counts
epochs = 100 

model = create_model()
if torch.cuda.is_available():
    model = model.cuda()


train_model(model, train_input, train_target, batch_size, epochs)
# save model
torch.save(model.state_dict(), 'net_2.pt')

nb_errors = compute_nb_errors(model, test_input, test_target, batch_size)

print ('error: {:.02f}% {:d} / {:d}'.format(100*nb_errors / test_input.size(0), nb_errors, test_input.size(0)))

