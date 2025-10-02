from torch import nn

class lenet(nn.Module): #Lenet神经网络
    def __init__(self):
        super(lenet , self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16,  kernel_size=5),  # input[3, 120, 120]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output[48, 27, 27]
            nn.Conv2d(16, 32, kernel_size=5),  # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output[128, 13, 13]
            nn.Flatten(),
            nn.Linear(23328, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 298),
        )
    def forward(self , x):
        x = self.model(x)
        return x