import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        feature_shared = self.layer3(x)
        feature_cls = self.layer4(feature_shared)
        #
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return feature_shared,feature_cls


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict("/home/yangwf/.torch/models/resnet101-5d3b4d8f.pth")
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load('/home/yangwf/.torch/models/resnet50-19c8e357.pth'),strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23,3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load("/home/yangwf/.torch/models/resnet101-5d3b4d8f.pth"))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


# class AttributeClassification(nn.Module):
#
#     def __init__(self):
#         super(AttributeClassification, self).__init__()
#         self.attribute_class_number = [6, 8, 5, 5, 5, 10, 6, 9]
#         self.main_net=resnet50(pretrained=True)
#         # self.basicblock1=BasicBlock(1024,2048,stride=2,downsample=nn.Sequential(nn.Conv2d(1024, 2048,
#         #                   kernel_size=1, stride=2, bias=False),nn.BatchNorm2d(2048),))
#         self.basicblock1=BasicBlock(1024,2048,stride=2,downsample=nn.Sequential(nn.Conv2d(1024, 2048,
#                           kernel_size=1, stride=2, bias=False),nn.BatchNorm2d(2048),))
#         self.basicblock2=BasicBlock(2048,2048)
#         self.basicblock3=BasicBlock(2048,2048)
#         #self.cls_MainNet=nn.Conv2d(2048,kernel_size=1,out_channels=self.number_attributes)
#
#         self.attribute1_conv = nn.Conv2d(2048,2048,kernel_size=3,padding=1)
#         self.attribute1_fc = nn.Linear(2048,self.attribute_class_number[0])
#
#         self.attribute2_conv = nn.Conv2d(2048,2048,kernel_size=3,padding=1)
#         self.attribute2_fc = nn.Linear(2048,self.attribute_class_number[1])
#
#         self.attribute3_conv = nn.Conv2d(2048,2048,kernel_size=3,padding=1)
#         self.attribute3_fc = nn.Linear(2048,self.attribute_class_number[2])
#
#         self.attribute4_conv = nn.Conv2d(2048,2048,kernel_size=3,padding=1)
#         self.attribute4_fc = nn.Linear(2048,self.attribute_class_number[3])
#
#         self.attribute5_conv = nn.Conv2d(2048,2048,kernel_size=3,padding=1)
#         self.attribute5_fc = nn.Linear(2048,self.attribute_class_number[4])
#
#         self.attribute6_conv = nn.Conv2d(2048,2048,kernel_size=3,padding=1)
#         self.attribute6_fc = nn.Linear(2048, self.attribute_class_number[5])
#
#         self.attribute7_conv = nn.Conv2d(2048,2048,kernel_size=3,padding=1)
#         self.attribute7_fc = nn.Linear(2048, self.attribute_class_number[6])
#
#         self.attribute8_conv = nn.Conv2d(2048,2048,kernel_size=3,padding=1)
#         self.attribute8_fc = nn.Linear(2048, self.attribute_class_number[7])
#
#
#
#
#
#         # self.attribute1_conv = nn.Conv2d(2048,1024,kernel_size=3,padding=1)
#         # self.attribute1_fc = nn.Linear(1024,self.attribute_class_number[0])
#         #
#         # self.attribute2_conv = nn.Conv2d(2048,1024,kernel_size=3,padding=1)
#         # self.attribute2_fc = nn.Linear(1024,self.attribute_class_number[1])
#         #
#         #self.attribute3_conv = nn.Conv2d(2048,1024,kernel_size=3,padding=1)
#         #self.attribute3_fc = nn.Linear(1024,self.attribute_class_number[2])
#         #self.attribute3 = nn.Sequential(nn.Conv2d(2048,1024,kernel_size=3,padding=1),)
#         #
#         # self.attribute4_conv = nn.Conv2d(2048,1024,kernel_size=3,padding=1)
#         # self.attribute4_fc = nn.Linear(1024,self.attribute_class_number[3])
#         #
#         # self.attribute5_conv = nn.Conv2d(2048,1024,kernel_size=3,padding=1)
#         # self.attribute5_fc = nn.Linear(1024,self.attribute_class_number[4])
#         #
#         # self.attribute6_conv = nn.Conv2d(2048,1024,kernel_size=3,padding=1)
#         # self.attribute6_fc = nn.Linear(1024, self.attribute_class_number[5])
#         #
#         # self.attribute7_conv = nn.Conv2d(2048,1024,kernel_size=3,padding=1)
#         # self.attribute7_fc = nn.Linear(1024, self.attribute_class_number[6])
#         #
#         # self.attribute8_conv = nn.Conv2d(2048,1024,kernel_size=3,padding=1)
#         # self.attribute8_fc = nn.Linear(1024, self.attribute_class_number[7])
#         #
#         #
#         #
#         #
#         #
#         self.relu=nn.ReLU()
#         self.avgpool=nn.AvgPool2d(kernel_size=7,stride=1)
#
#     def weight_init(self,params):
#         own_dict=self.main_net.state_dict()
#         for name,val in own_dict.items():
#             val.copy_(params[name])
#
#     def set_finetune(self):
#         for parameter in self.main_net.parameters():
#             parameter.require_grad=False
#
#
#     def forward(self,x):
#         feature=self.main_net(x)
#         #print feature.size()
#         feature_MainBranch=self.basicblock1(feature)
#         feature_MainBranch=self.basicblock2(feature_MainBranch)
#         feature_MainBranch=self.basicblock3(feature_MainBranch)
#
#         attribute1_predict=self.attribute1_conv(feature_MainBranch)
#         attribute1_predict=self.relu(attribute1_predict)
#         attribute1_predict=self.avgpool(attribute1_predict)
#         #attribute1_predict=attribute1_predict.view(-1,2048)
#         attribute1_predict = attribute1_predict.view(-1, 2048)
#         attribute1_predict=self.attribute1_fc(attribute1_predict)
#         #attribute1_predict=torch.nn.Softmax(attribute1_predict)
#
#         attribute2_predict=self.attribute2_conv(feature_MainBranch)
#         attribute2_predict=self.relu(attribute2_predict)
#         attribute2_predict=self.avgpool(attribute2_predict)
#         #attribute2_predict=attribute2_predict.view(-1,2048)
#         attribute2_predict = attribute2_predict.view(-1, 2048)
#         attribute2_predict=self.attribute2_fc(attribute2_predict)
#         #attribute2_predict=torch.nn.Softmax(attribute2_predict)
#
#         attribute3_predict=self.attribute3_conv(feature_MainBranch)
#         attribute3_predict=self.relu(attribute3_predict)
#         attribute3_predict=self.avgpool(attribute3_predict)
#         # #attribute3_predict=attribute3_predict.view(-1,2048)
#         attribute3_predict=attribute3_predict.view(-1,2048)
#         attribute3_predict=self.attribute3_fc(attribute3_predict)
#         #attribute3_predict=torch.nn.Softmax(attribute3_predict)
#         #feature=feature.view(-1,2048)
#         #attribute3_predict = self.attribute3_fc(attribute3_predict)
#
#         attribute4_predict=self.attribute4_conv(feature_MainBranch)
#         attribute4_predict=self.relu(attribute4_predict)
#         attribute4_predict=self.avgpool(attribute4_predict)
#         #attribute4_predict=attribute4_predict.view(-1,2048)
#         attribute4_predict = attribute4_predict.view(-1, 2048)
#         attribute4_predict=self.attribute4_fc(attribute4_predict)
#         #attribute4_predict=torch.nn.Softmax(attribute4_predict)
#
#         attribute5_predict=self.attribute5_conv(feature_MainBranch)
#         attribute5_predict=self.relu(attribute5_predict)
#         attribute5_predict=self.avgpool(attribute5_predict)
#         #attribute5_predict=attribute5_predict.view(-1,2048)
#         attribute5_predict = attribute5_predict.view(-1, 2048)
#         attribute5_predict=self.attribute5_fc(attribute5_predict)
#         #attribute5_predict=torch.nn.Softmax(attribute5_predict)
#
#         attribute6_predict=self.attribute6_conv(feature_MainBranch)
#         attribute6_predict=self.relu(attribute6_predict)
#         attribute6_predict=self.avgpool(attribute6_predict)
#         #attribute6_predict=attribute6_predict.view(-1,2048)
#         attribute6_predict = attribute6_predict.view(-1, 2048)
#         attribute6_predict=self.attribute6_fc(attribute6_predict)
#         #attribute6_predict=torch.nn.Softmax(attribute6_predict)
#
#         attribute7_predict=self.attribute7_conv(feature_MainBranch)
#         attribute7_predict=self.relu(attribute7_predict)
#         attribute7_predict=self.avgpool(attribute7_predict)
#         #attribute7_predict=attribute7_predict.view(-1,2048)
#         attribute7_predict = attribute7_predict.view(-1, 1024)
#         attribute7_predict=self.attribute7_fc(attribute7_predict)
#         #attribute7_predict=torch.nn.Softmax(attribute7_predict)
#
#         attribute8_predict=self.attribute8_conv(feature_MainBranch)
#         attribute8_predict=self.relu(attribute8_predict)
#         attribute8_predict=self.avgpool(attribute8_predict)
#         #attribute8_predict=attribute8_predict.view(-1,2048)
#         attribute8_predict = attribute8_predict.view(-1, 2048)
#         attribute8_predict=self.attribute8_fc(attribute8_predict)
#         #attribute8_predict=torch.nn.Softmax(attribute8_predict)
#
#         return attribute1_predict,attribute2_predict,attribute3_predict,attribute4_predict,attribute5_predict,attribute6_predict,attribute7_predict,attribute8_predict
class Reshape(nn.Module):
    def __init__(self, args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class AttributeClassification(nn.Module):
    def __init__(self,pretrained=False):
        self.class_number=[6,8,5,5,5,10,6,9]
        super(AttributeClassification, self).__init__()
        self.MainNet=resnet50(pretrained=pretrained)




        self.conv0 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1)

        self.fc0 = nn.Linear(2048,self.class_number[0])
        self.fc1 = nn.Linear(2048, self.class_number[1])
        self.fc2 = nn.Linear(2048, self.class_number[2])
        self.fc3 = nn.Linear(2048, self.class_number[3])
        self.fc4 = nn.Linear(2048, self.class_number[4])
        self.fc5 = nn.Linear(2048, self.class_number[5])
        self.fc6 = nn.Linear(2048, self.class_number[6])
        self.fc7 = nn.Linear(2048, self.class_number[7])
        self.avgpool=nn.AvgPool2d(kernel_size=7,stride=1)

    def forward(self, x):
        featured_shared,feature_main=self.MainNet(x)

        feature_cls0 = feature_main
        feature_cls1 = feature_main
        feature_cls2 = feature_main
        feature_cls3 = feature_main
        feature_cls4 = feature_main
        feature_cls5 = feature_main
        feature_cls6 = feature_main
        feature_cls7 = feature_main

        # feature_cls0 = self.conv0(feature_main)
        # feature_cls1 = self.conv1(feature_main)
        # feature_cls2 = self.conv2(feature_main)
        # feature_cls3 = self.conv3(feature_main)
        # feature_cls4 = self.conv4(feature_main)
        # feature_cls5 = self.conv5(feature_main)
        # feature_cls6 = self.conv6(feature_main)
        # feature_cls7 = self.conv7(feature_main)

        feature_final0 = self.avgpool(feature_cls0).view(-1,2048)
        feature_final1 = self.avgpool(feature_cls1).view(-1,2048)
        feature_final2 = self.avgpool(feature_cls2).view(-1,2048)
        feature_final3 = self.avgpool(feature_cls3).view(-1,2048)
        feature_final4 = self.avgpool(feature_cls4).view(-1,2048)
        feature_final5 = self.avgpool(feature_cls5).view(-1,2048)
        feature_final6 = self.avgpool(feature_cls6).view(-1,2048)
        feature_final7 = self.avgpool(feature_cls7).view(-1,2048)

        feature_final0 = nn.Dropout()(feature_final0)
        feature_final1 = nn.Dropout()(feature_final1)
        feature_final2 = nn.Dropout()(feature_final2)
        feature_final3 = nn.Dropout()(feature_final3)
        feature_final4 = nn.Dropout()(feature_final4)
        feature_final5 = nn.Dropout()(feature_final5)
        feature_final6 = nn.Dropout()(feature_final6)
        feature_final7 = nn.Dropout()(feature_final7)

        predict0 = self.fc0(feature_final0)
        predict1 = self.fc1(feature_final1)
        predict2 = self.fc2(feature_final2)
        predict3 = self.fc3(feature_final3)
        predict4 = self.fc4(feature_final4)
        predict5 = self.fc5(feature_final5)
        predict6 = self.fc6(feature_final6)
        predict7 = self.fc7(feature_final7)

        predict0 = nn.Softmax(dim=1)(predict0)
        predict1 = nn.Softmax(dim=1)(predict1)
        predict2 = nn.Softmax(dim=1)(predict2)
        predict3 = nn.Softmax(dim=1)(predict3)
        predict4 = nn.Softmax(dim=1)(predict4)
        predict5 = nn.Softmax(dim=1)(predict5)
        predict6 = nn.Softmax(dim=1)(predict6)
        predict7 = nn.Softmax(dim=1)(predict7)
        #predict=torch.cat((predict0,predict1,predict2,predict3,predict4,predict5,predict6,predict7),dim=1)
        return predict0,predict1,predict2,predict3,predict4,predict5,predict6,predict7
        #return predict

