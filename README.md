# Number-Neural-Net

Simple 2d convolution neural network that utilizes pytorch to figure out the numbers in the MNIST dataset.

The neural network is described by class Net and it features 2 Convolutional layers followed by two fully connected linear layers. 
The first convolution uses ReLU then has dropout and is followed by max pooling. 
The second convolution layer also uses ReLU then has dropout and is followed by max pooling.
The first linear layer uses ReLU and then dropout.
The last linear layer uses a log softmax function.

```Py
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.Dropout2d(),                      
            nn.MaxPool2d(2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=10, out_channels=32, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.Dropout2d(),                      
            nn.MaxPool2d(2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1568, 50),
                nn.ReLU(),
                nn.Dropout(),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(50, 10),
            nn.LogSoftmax(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        output = self.fc2(x)
        return output
```
