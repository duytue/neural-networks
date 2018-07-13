# Net comparision
## Configuration
* Models are trained with:
    * NVIDIA Tesla K80
    * CUDA 8.0
    * PyTorch 0.4

## Overview
Parameters|net.pt|net_2.pt|net_3.pt|net_4.pt|net_5.pt|net_6.pt
---|---|---|---|---|---|---
batch_size|50|100|100|100|100|100
epochs|250|100|75(50)|50|150|250
eta|1e-3|1e-2|1e-2|1e-2|1e-2|1e-2
training_time|0:44|0:30|0:20|0:15|0:40|1:40
initial_error| > .032|.021|N/A|N/A|20.314|N/A
train_acc|84%|84.992%|99.128%|100%|88.228%|91.150%
test_error|44%|36.69%|28.74%|27.32%|27.40%|26.72%

## Additional Information
* net_1
    * Base CNN with 15 convolutional layers, 2 FC and 1 Softmax layer
    ```python 
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
    ```
* net_2
    * Different batch size, epochs and eta
* net_3
    * Xavier initialization for FC -> Faster training (28 epochs to reach 98% train_acc)
    * NO SOFTMAX LAYER -> Important for high accuracy
    ```python
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
        
    model.apply(init_weights)
    ```
* net_4
    * Add batch normalization for FC: faster training time, lower epoch size, test_acc increases by 1%.
    ```python 
    def create_classifier():
        return [Flatten(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 1024),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 10)]
    ```
* net_5
    * Swap batch norm for drop out: Accuracy and test error increases.
    * Xavier
    ```python 
    def create_classifier():
        return [Flatten(),
        nn.Dropout(p=0.5),
        nn.Linear(1024, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(1024, 10)]
    ```

* net_6
    * Drop out
    * Xavier
    * increase epochs to 250