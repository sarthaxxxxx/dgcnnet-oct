import torch 
import torch.nn as nn
import torch.nn.functional as F

from bottleneck import Bottleneck as bot


def DGCNNet_ResNet101(classes):
    return ResNet(bot, [3, 4, 23, 3], classes)


def DGCNNet_ResNet50(classes):
    return ResNet(bot,  [3, 4, 6, 3], classes)

class ResNet(nn.Module):
    def __init__(self, blocks, 
                layers, 
                classes):
        super(ResNet, self).__init__()
        self.inplane = 128
        # 3x3 conv w/ padding
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias = False), 
        nn.BatchNorm2d(64), 
        nn.ReLU(inplace = True), 
        nn.Conv2d(64, 64, 3, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace = True),
        nn.Conv2d(64, 128, 3, 1))
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.bn = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace = False)
        self.maxpool = nn.MaxPool2d(3, 2, 1, ceil_mode = True) #use ceil instead of the default floor, during convolution.
        self.layer1 = self.__make_layers__(blocks, 64, layers[0])
        self.layer2 = self.__make_layers__(blocks, 128, layers[1], stride = 2)
        self.layer3 = self.__make_layers__(blocks, 256, layers[2], dilation = 2)
        self.layer4 = self.__make_layers__(blocks, 512, layers[3], dilation = 4, multi_grid = (1, 2, 4))

        #DualGCN
        self.head = DualGCNHead(2048, 512, classes)
        
        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.Dropout(0.1),
            nn.Conv2d(512, classes, 1, 1, bias = True)
        )

    def forward(self, a):
        a = self.conv1(a)
        a = self.bn(a)
        a = self.relu(a)
        a = self.maxpool(a)
        a = self.layer1(a)
        a = self.layer2(a)
        a = self.layer3(a)
        if self.training:
            a_dsn = self.dsn(a)
        a = self.layer4(a)
        a = self.head(a)
        if self.training: return [a, a_dsn]
        else: return a


    def __make_layers__(self, block, planes, blocks, stride = 1, dilation = 1, multi_grid = 1):
        ds = None
        if stride != 1 or self.inplane != block.expansion * planes:
           ds = nn.Sequential(nn.Conv2d(self.inplane, block.expansion * planes,
                              kernel_size = 1, stride = stride, bias = False),
                              nn.BatchNorm2d(block.expansion * planes))
        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1

        layers.append(block(self.inplane, planes, stride, dilation, ds, generate_multi_grid(0, multi_grid)))

        self.inplane = block.expansion * planes

        for idx in range(1, blocks):
            layers.append(block(self.inplane, planes, dilation = dilation, 
                                downsample = ds, 
                                multi_grid = generate_multi_grid(idx, multi_grid)))

        return nn.Sequential(*layers)


class CoordinateGCN(nn.Module):
    def __init__(self, plane):
        super(CoordinateGCN, self).__init__()
        inter_plane = plane // 2 #(d -> downsampling operation)
        self.node_delta = nn.Conv2d(plane, inter_plane, 1)
        self.node_psi = nn.Conv2d(plane, inter_plane, 1)
        self.node_vu = nn.Conv2d(plane, inter_plane, 1)
        self.conv_ws = nn.Conv1d(inter_plane, inter_plane, 1, bias = False)
        self.bn_ws = nn.BatchNorm1d(inter_plane)
        self.soft = nn.Softmax(dim = 2)
        self.out = nn.Sequential(nn.Conv2d(inter_plane, plane, 1),
                                 nn.BatchNorm2d(plane))

    def forward(self, a):
        node_delta = self.node_delta(a)
        node_psi = self.node_psi(a)
        node_vu = self.node_vu(a)

        batch, channels, h, w = node_delta.size()
        node_delta = node_delta.view(batch, channels, -1).permute(0, 2, 1)
        node_psi = node_psi.view(batch, channels, -1)
        node_vu = node_vu.view(batch, channels, -1).permute(0, 2, 1)

        intermediate = self.soft(torch.bmm(node_psi, node_vu))  #bmm between (b,c,h*w) and (b, h*w, c) ---> (b,c,c)
        intermediate = torch.bmm(node_delta, intermediate) #bmm between (b, h*w, c) and (b,c,c) ---> (b,h*w,c)
        intermediate = intermediate.transpose(1, 2).contiguous() #(b, c, h*w)

        AVW = self.bn_ws(self.conv_ws(intermediate))
        AVW = AVW.view(batch, channels, h, -1)

        return F.relu_(self.out(AVW) + a) #add input feature w/ the coordinate space feature. 



class DualGCN(nn.Module):
    def __init__(self, planes, ratio = 4):
        super(DualGCN, self).__init__()
        self.phi = nn.Conv2d(planes, planes // ratio * 2, 1, bias = False) # planes // ratio * 2 : D1 (D/2)
        self.phi_bn = nn.BatchNorm2d(planes // ratio * 2)
        self.theta = nn.Conv2d(planes, planes // ratio, 1, bias = False) # planes // ratio : D2 (D/4)
        self.theta_bn = nn.BatchNorm2d(planes // ratio)

        #Adjacency Matric (A_F)
        self.conv_adj = nn.Conv1d(planes // ratio, planes // ratio, 1, bias = False)
        self.bn_adj = nn.BatchNorm1d(planes // ratio)

        #State update function (weight matrix (W_F))
        self.conv_w = nn.Conv1d(planes // ratio * 2, planes // ratio * 2, 1, bias = False)
        self.bn_w = nn.BatchNorm1d(planes // ratio * 2)

        # last f_c layer
        self.conv = nn.Conv2d(planes // ratio * 2, planes, 1, bias = False)
        self.bn = nn.BatchNorm2d(planes)

        self.local_network = nn.Sequential(
            nn.Conv2d(planes, planes, 3, stride = 2, padding = 1, bias = False, groups = planes),
            nn.BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, stride = 2, padding = 1, bias = False, groups = planes),
            nn.BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, stride = 2, padding = 1, bias = False, groups = planes),
            nn.BatchNorm2d(planes)
        )

        self.cgn = CoordinateGCN(planes)
        self.final = nn.Sequential(nn.Conv2d(planes *2, planes, 1, bias = False),
                                   nn.BatchNorm2d(planes))

    def forward(self, a):
        '''
        Local space
        '''
        x = a
        loc = self.local_network(a) #not sure what this is.

        loc = self.cgn(loc) #from cooridnate space GCN.
        loc = F.interpolate(loc, size = x.size()[2:],               # why is it done here?
                            mode = 'bilinear', align_corners = True)
        coordinate_feat_out = (x * loc) + x

        '''
        Feature space projection
        '''
        x_a, x_b = x , x

        x_a = self.reshape_tensor(self.phi_bn(self.phi(x_a))) # (b,c,h*w)
        x_b = self.reshape_tensor(self.theta_bn(self.theta(x_b))) # (b,c,h*w)

        intermediate = torch.matmul(x_a, x_b.transpose(1, 2))

        '''
        Graph interaction 
        '''
        z = intermediate.transpose(1, 2).contiguous()
        z = self.conv_adj(intermediate)
        z = self.bn_adj(z)
        z = z.transpose(1, 2).contiguous()

        z += intermediate #add graph output with projection feature

        wf = self.conv_w(z)
        wf = self.bn_w(z)

        df = torch.matmul(wf, x_b)

        batch, chan, h, w = x.size()
        y = df.view(batch, -1, h, w)

        y = self.conv(y)
        y = self.bn(y)

        '''
        Reprojection
        '''
        int_out = F.relu_(y + x)

        return self.final(torch.cat([int_out, coordinate_feat_out], 1)) # along the channels
        
    def reshape_tensor(self, mat):
        b, c , h, w = mat.size()
        return mat.view(b, c, -1)


class DualGCNHead(nn.Module):
    def __init__(self, in_planes, interplanes, classes):
        super(DualGCNHead, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, interplanes, 3,
                                   padding = 1, bias = False),
                                   nn.BatchNorm2d(interplanes),
                                   nn.ReLU(interplanes))
        self.dualgcn = DualGCN(interplanes)
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes + interplanes, interplanes, 3,
                                   padding = 1, dilation = 1, bias = False),
                                   nn.BatchNorm2d(interplanes),
                                   nn.ReLU(interplanes))                           
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_planes + interplanes, interplanes, 3, padding = 1,
                      dilation = 1, bias = False),
            nn.BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
            nn.Conv2d(512, classes, 1, stride = 1, padding = 0, bias = True))

    def forward(self, a):
        out = self.conv1(a)
        out = self.dualgcn(a)
        out = self.conv2(a)
        out = self.bottleneck(torch.cat([out, a], 1))
        return out