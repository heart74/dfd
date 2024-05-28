import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from backbone.components.attention import SFIA, ChannelAttention, SE, CAT, CMIA 
from backbone.components.freqency import FMFM, BlockDCT, TEM_CONV

pretrained_paths = {
    'efficientnet_b4': '/media/sdd/zhy/paper/model_zoo/efficientnet_b4_ra2_320-7eb33cd5.pth',
    'efficientnet_b2': '/media/sdd/zhy/paper/model_zoo/efficientnet_b2_ra-bcdf34b7.pth',
    'efficientnet_b0': '/media/sdd/zhy/paper/model_zoo/efficientnet_b0_ra-3dd342df.pth',
}

CMIA_cins = {
    'efficientnet_b4': [272,448],
    'efficientnet_b2': [208,352],
    'efficientnet_b0': [192,320]
}
CMIA_dims = {
    'efficientnet_b4': [100,100],
    'efficientnet_b2': [64,64],
    'efficientnet_b0': [49,49]
}

input_size = {
    'efficientnet_b4': (3, 320, 320),
    'efficientnet_b2': (3, 256, 256),
    'efficientnet_b0': (3, 224, 224)
}

class CUS_EFF(nn.Module):
    def __init__(self, model_name='efficientnet_b0', pretrained=False, **kwargs):
        super(CUS_EFF, self).__init__()
        # 加载预训练的EfficientNet模型
        # pretrained_cfg = timm.models.create_model(model_name).default_cfg
        # pretrained_cfg['file']=pretrained_paths[model_name]
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=2,  **kwargs)

        # 添加自定义属性
        self.stem_layer = nn.Sequential(
            *list(self.model.children())[:2]
        )

        self.before_pool = nn.Sequential(
            *list(self.model.children())[3:5]
        )

        self.before_fc = nn.Sequential(
            *list(self.model.children())[5:-1]
        )
        # 1,2,3,4
        for i in range(1,5):
            setattr(self, f'fea_{i}', nn.Sequential(*list(self.model.children())[2][2*(i-1):2*i]))

    def forward_stem(self, x):
        return self.stem_layer(x)
    
    def forward_feas(self, x):
        x = self.fea_1(x)
        x = self.fea_2(x)
        x = self.fea_3(x)
        x = self.fea_4(x)
        return x
    
    def forward_before_pool(self, x):
        return self.before_pool(x)

    def forward_before_fc(self, x):
        return self.before_fc(x)
    
    def forward_fc(self,x):
        return self.model.classifier(x)

    def forward(self, x):
        x = self.forward_stem(x)
        x = self.forward_feas(x)
        x = self.forward_before_pool(x)
        x = self.forward_before_fc(x)
        x = self.forward_fc(x)
        return x


class EFF_CMIA(nn.Module):
    def __init__(self, model_name='efficientnet_b0', pretrained=False, embddings=64, freq_srm=(0,0),fusion="ca",**kwargs):
        super(EFF_CMIA, self).__init__()
        self.model_spa = CUS_EFF(model_name, pretrained, **kwargs)
        self.model_freq = CUS_EFF(model_name, pretrained, **kwargs)
        cins = CMIA_cins[model_name]
        dims = CMIA_dims[model_name]
        self.cmia0 = CMIA(cins[0], dims[0])
        self.cmia1 = CMIA(cins[1], dims[1])

        freq,srm = freq_srm
        if srm==1:
            self.tem = TEM_CONV(outc=3)
        else:
            self.tem = nn.Identity()
        if freq==1:
            self.blockDCT = nn.Sequential(
                BlockDCT(max(input_size[model_name])),
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, bias=False)
            )
        elif freq==2:
            self.blockDCT = nn.Sequential(
                BlockDCT(8),
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, bias=False)
            )
        elif freq==3:
            self.blockDCT = FMFM(in_channels=3,out_channels=3)
        else:
            self.blockDCT = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, bias=False)

        num_features = self.model_spa.model.num_features

        if embddings!=0:
            self.down = nn.Linear(num_features, embddings)
            self.model_spa.model.classifier = nn.Linear(embddings,2)
            self.channel = embddings
        else:
            self.down = nn.Identity()
            self.channel = num_features

        # To Do: fusion
        self.fusion = SE(inc=num_features*2,ouc=num_features)

    def preprocess(self, x):
        _,C,_,_ = x.size()
        spa, freq = x[:,:C//2,:,:], x[:,C//2:,:,:]
        spa = self.tem(spa)
        freq = self.blockDCT(freq)
        return spa, freq

    def get_embedding(self, x):
        x_spa, x_freq = self.preprocess(x)

        stem_spa = self.model_spa.forward_stem(x_spa)
        stem_freq = self.model_freq.forward_stem(x_freq)

        fea_spa = self.model_spa.fea_1(stem_spa)
        fea_freq = self.model_freq.fea_1(stem_freq)

        fea_spa = self.model_spa.fea_2(fea_spa)
        fea_freq = self.model_freq.fea_2(fea_freq)

        fea_spa = self.model_spa.fea_3(fea_spa)
        fea_freq = self.model_freq.fea_3(fea_freq)
        fea_spa,fea_freq = self.cmia0(fea_spa, fea_freq)

        fea_spa = self.model_spa.fea_4(fea_spa)
        fea_freq = self.model_freq.fea_4(fea_freq)
        fea_spa,fea_freq = self.cmia1(fea_spa, fea_freq)

        before_pool_spa = self.model_spa.forward_before_pool(fea_spa)
        before_pool_freq = self.model_freq.forward_before_pool(fea_freq)

        fea = self.fusion(before_pool_spa, before_pool_freq)
        fea = self.down(self.model_spa.model.global_pool(fea))
        fea = F.normalize(fea, dim=1, p=2) # map to the unit sphere
        return fea

    def forward(self, x):
        fea = self.get_embedding(x)
        out = self.model_spa.model.classifier(fea)
        return fea, out


class EFF_woCMIA(nn.Module):
    def __init__(self, model_name='efficientnet_b0', pretrained=True, embddings=64, freq_srm=(0,0),fusion="ca",**kwargs):
        super(EFF_woCMIA, self).__init__()
        self.model_spa = CUS_EFF(model_name, pretrained, **kwargs)
        self.model_freq = CUS_EFF(model_name, pretrained, **kwargs)
        cins = CMIA_cins[model_name]
        dims = CMIA_dims[model_name]
        # self.cmia0 = CMIA(cins[0], dims[0])
        self.cmia1 = CMIA(cins[1], dims[1])

        freq,srm = freq_srm
        if srm==1:
            self.tem = TEM_CONV(outc=3)
        else:
            self.tem = nn.Identity()
        if freq==1:
            self.blockDCT = nn.Sequential(
                BlockDCT(max(input_size[model_name])),
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, bias=False)
            )
        elif freq==2:
            self.blockDCT = nn.Sequential(
                BlockDCT(8),
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, bias=False)
            )
        elif freq==3:
            self.blockDCT = FMFM(in_channels=3,out_channels=3)
        else:
            self.blockDCT = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, bias=False)

        num_features = self.model_spa.model.num_features

        if embddings!=0:
            self.down = nn.Linear(num_features, embddings)
            self.model_spa.model.classifier = nn.Linear(embddings,2)
            self.channel = embddings
        else:
            self.down = nn.Identity()
            self.channel = num_features

        # To Do: fusion
        self.fusion = SE(inc=num_features*2,ouc=num_features)

    def preprocess(self, x):
        _,C,_,_ = x.size()
        spa, freq = x[:,:C//2,:,:], x[:,C//2:,:,:]
        spa = self.tem(spa)
        freq = self.blockDCT(freq)
        return spa, freq

    def get_embedding(self, x):
        x_spa, x_freq = self.preprocess(x)

        stem_spa = self.model_spa.forward_stem(x_spa)
        stem_freq = self.model_freq.forward_stem(x_freq)

        fea_spa = self.model_spa.fea_1(stem_spa)
        fea_freq = self.model_freq.fea_1(stem_freq)

        fea_spa = self.model_spa.fea_2(fea_spa)
        fea_freq = self.model_freq.fea_2(fea_freq)

        fea_spa = self.model_spa.fea_3(fea_spa)
        fea_freq = self.model_freq.fea_3(fea_freq)
        # fea_spa,fea_freq = self.cmia0(fea_spa, fea_freq)

        fea_spa = self.model_spa.fea_4(fea_spa)
        fea_freq = self.model_freq.fea_4(fea_freq)
        fea_spa,fea_freq = self.cmia1(fea_spa, fea_freq)

        before_pool_spa = self.model_spa.forward_before_pool(fea_spa)
        before_pool_freq = self.model_freq.forward_before_pool(fea_freq)

        fea = self.fusion(before_pool_spa, before_pool_freq)
        fea = self.down(self.model_spa.model.global_pool(fea))
        # fea = F.normalize(fea, dim=1, p=2) # map to the unit sphere
        return fea

    def forward(self, x):
        fea = self.get_embedding(x)
        out = self.model_spa.model.classifier(fea)
        return out