import torch
import torch.nn as nn
import torch.nn.functional as F


class Net_1(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # Input block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )
        # Convolutional Block 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3,3), padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(0.1)
        )
        
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3,3), padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1)
        )
        self.pool1 = nn.MaxPool2d(2,2)
        # Transition Block 1
        self.transition1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1,1), padding=0, stride=1))
        
        
        # Convolutional Block 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3,3), padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1)
        )
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3,3), padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(0.1)
        )
        self.transition2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=24, kernel_size=(1,1), padding=0, stride=1))
        # Output block
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=18, kernel_size=(3,3), padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(18),
            nn.Dropout(0.1)
        )
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=10, kernel_size=(3,3), padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.1)
        )
        
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1,1))
        )
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1,1), padding=0, stride=1)
        )
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.transition1(x)
        
        
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.transition2(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        
        x = self.gap(x)
        x = self.convblock9(x)
        
        # Flatten the output to match the expected shape
        x = x.view(x.size(0), -1)  # Flatten the output while keeping the batch size
        return F.log_softmax(x, dim=-1)
    
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # Input block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.15)
        )
        # Convolutional Block 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3,3), padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(0.15)
        )
        
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3,3), padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.15)
        )
        self.pool1 = nn.MaxPool2d(2,2)
        # Transition Block 1
        self.transition1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1,1), padding=0, stride=1))
        
        
        # Convolutional Block 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3,3), padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.15)
        )
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.15)
        )
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3,3), padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(0.15)
        )
        self.transition2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=24, kernel_size=(1,1), padding=0, stride=1))
        # Output block
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=18, kernel_size=(3,3), padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(18),
            nn.Dropout(0.15)
        )
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=10, kernel_size=(3,3), padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.15)
        )
        
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1,1))
        )
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1,1), padding=0, stride=1)
        )
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.transition1(x)
        
        
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.transition2(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        
        x = self.gap(x)
        x = self.convblock9(x)
        
        # Flatten the output to match the expected shape
        x = x.view(x.size(0), -1)  # Flatten the output while keeping the batch size
        return F.log_softmax(x, dim=-1)

