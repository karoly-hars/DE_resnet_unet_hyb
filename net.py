import torch.nn as nn
import torch
import os


__all__ = ['hyb_net']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
      


def get_incoming_shape(incoming):
    """ Returns the incoming data shape """
    size = incoming.size()
    return [size[0], size[1], size[2], size[3]]

def interleave(tensors, axis):
    # change the first element (batch_size to -1)
    old_shape = get_incoming_shape(tensors[0])[1:]
    new_shape = [-1] + old_shape
    
    # double 1 dimension
    new_shape[axis] *= len(tensors)
    
    # pack the tensors on top of each other 
    stacked = torch.stack(tensors, axis+1)
    
    # reshape and return
    reshaped = stacked.view(new_shape)
    return reshaped


        
class UnPool_as_Conv(nn.Module):       
    def __init__(self, inplanes, planes):
        super(UnPool_as_Conv, self).__init__()
        
        # interleaving convolutions
        self.conv_A = nn.Conv2d(in_channels = inplanes, out_channels = planes, kernel_size = (3,3), stride = 1, padding = 1)
        self.conv_B = nn.Conv2d(in_channels = inplanes, out_channels = planes, kernel_size = (2,3), stride = 1, padding = 0)
        self.conv_C = nn.Conv2d(in_channels = inplanes, out_channels = planes, kernel_size = (3,2), stride = 1, padding = 0)
        self.conv_D = nn.Conv2d(in_channels = inplanes, out_channels = planes, kernel_size = (2,2), stride = 1, padding = 0)
        
        
    def forward(self,x):
        output_A = self.conv_A(x)
        
        padded_B = nn.functional.pad(x, (1,1,0,1))
        output_B = self.conv_B(padded_B)
        
        padded_C = nn.functional.pad(x, (0,1,1,1))
        output_C = self.conv_C(padded_C)
        
        padded_D = nn.functional.pad(x, (0,1,0,1))
        output_D = self.conv_D(padded_D)       
        
        left = interleave([output_A, output_B], axis = 2)
        right = interleave([output_C, output_D], axis = 2)
        Y = interleave([left, right], axis = 3)
                
        return Y



class UpProj(nn.Module):  
    def __init__(self, inplanes, planes):
        super(UpProj, self).__init__()        
        
        self.unpool_main = UnPool_as_Conv(inplanes, planes)
        self.unpool_res = UnPool_as_Conv(inplanes, planes)
             
        self.main_branch = nn.Sequential(
            self.unpool_main,
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=False),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes)
        )
        
        self.residual_branch = nn.Sequential(
            self.unpool_res,
            nn.BatchNorm2d(planes),
        )

        self.relu = nn.ReLU(inplace=False)       
        
        
    def forward(self, input_data):
        
        x = self.main_branch(input_data)
        res = self.residual_branch(input_data)             
        x += res  
        x = self.relu(x)
        
        return x

class ConConv(nn.Module):
    def __init__(self, inplanes_x1, inplanes_x2, planes):
        super(ConConv, self).__init__()        
        self.conv = nn.Conv2d(inplanes_x1 + inplanes_x2, planes, kernel_size=1, bias=True)
    
    def forward(self, x1, x2):
        x1 = torch.cat([x2, x1], dim=1)
        x1 = self.conv(x1)
        return x1



class ResNetUpProjUnetHyb(nn.Module):
 
    def __init__(self, block, layers):
        
        self.inplanes = 64
        
        super(ResNetUpProjUnetHyb, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
               
        
        ''' additional up projection layers parts '''
        self.conv2 = nn.Conv2d(2048, 1024, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(1024)
        
        self.up_proj1 = UpProj(1024, 512)
        self.up_proj2 = UpProj(512, 256)
        self.up_proj3 = UpProj(256, 128)
        self.up_proj4 = UpProj(128, 64)
        
        self.drop = nn.Dropout(0.5, False)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True)
        
        ''' padding + concat for unet stuff '''
        self.con_conv1 = ConConv(1024, 512, 512)
        self.con_conv2 = ConConv(512, 256, 256) 
        self.con_conv3 = ConConv(256, 128, 128) 
        self.con_conv4 = ConConv(64, 64, 64) 
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                                
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x_to_conv4 = self.relu(x)
        
        x = self.maxpool(x_to_conv4)
        x_to_conv3 = self.layer1(x)
        x_to_conv2 = self.layer2(x_to_conv3)
        x_to_conv1 = self.layer3(x_to_conv2)
        x = self.layer4(x_to_conv1)
          
        ''' additional layers '''
        x = self.conv2(x)
        x = self.bn2(x)
        
        ''' up project part'''
        x = self.up_proj1(x)
        x = self.con_conv1(x, x_to_conv1)
        
        x = self.up_proj2(x)
        x = self.con_conv2(x, x_to_conv2)
        
        x = self.up_proj3(x)
        x = self.con_conv3(x, x_to_conv3)
        
        x = self.up_proj4(x)
        x = self.con_conv4(x, x_to_conv4)
        
        x = self.drop(x)
        x = self.conv3(x)
        x = self.relu(x)
        
        return x



def hyb_net(load_path='hyb_net_weights.model', use_gpu=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (int): 1 download model pretrained on ImageNet, 2 use previously saved model
    """
    model = ResNetUpProjUnetHyb(Bottleneck, [3, 4, 6, 3], **kwargs)
        
    if not os.path.exists(load_path):
        print('Downloading model weights...')
        os.system("curl https://transfer.sh/Htcjw/hyb_net_weights.model -o {}".format(load_path))
        print('Done.')
            
    if use_gpu:
        model.load_state_dict(torch.load(load_path))
    else:
        model.load_state_dict(torch.load(load_path, map_location='cpu'))

    
    return model
