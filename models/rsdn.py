from torch import nn


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(9,64,(1,5,5),stride=1,padding=5//2)
        #self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv3d(64, 48,(1,5,5),stride=1,padding=5//2)#56,60
        self.relu = nn.ReLU(inplace=True)
        self.pix=nn.PixelShuffle(4)

    def forward(self, x,start=0):
        print("1",x.shape)
        x=x.permute(0,2,1,3,4)
        x = self.relu(self.conv1(x))
        #x = self.relu(self.conv2(x))

        y = self.conv3(x)
        print("1", y.shape)
        y=y.permute(0,2,1,3,4)
        y=self.pix(y)
        print("1", y.shape)


        return y,y