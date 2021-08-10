import torch.nn as nn
import torch

class Focus(nn.Module):
    def __init__(self):
        super(Focus,self).__init__()
        self.conv = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,padding=1,stride=1)

    def forward(self,x):
        return self.conv( torch.cat([ x[...,::2,::2],x[...,1::2,::2],x[...,0::2,1::2],x[...,1::2,1::2] ], dim=1) )

if __name__=='__main__':
    with open('models/yolov5s.yaml') as f:
        import yaml
        yml = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        print()