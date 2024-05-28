"""
Author: Andreas Rössler
"""
import timm
from timm.data import resolve_data_config
from .two_eff import EFF_CMIA

pretrained_models = [
    'efficientnet_b0',
    'efficientnet_b2',
    'efficientnet_b4',
    # 'convnext_tiny',
]

pretrained_paths = {
    'efficientnet_b4': '/media/sdd/zhy/paper/model_zoo/efficientnet_b4_ra2_320-7eb33cd5.pth',
    'efficientnet_b2': '/media/sdd/zhy/paper/model_zoo/efficientnet_b2_ra-bcdf34b7.pth',
    'efficientnet_b0': '/media/sdd/zhy/paper/model_zoo/efficientnet_b0_ra-3dd342df.pth',
}

global_input_sizes = {
            'efficientnet_b0': (3, 224, 224),
            'efficientnet_b1': (3, 224, 224),
            'efficientnet_b2': (3, 256, 256),
            'efficientnet_b3': (3, 288, 288),
            'efficientnet_b4': (3, 320, 320),
        }

def model_selection(modelname, num_out_classes=2, use_fc=True, spa_inc=6, freq_srm=(0,0),input_size=(3,256,256), fusion='se', eff = 'efficientnet_b0',embeddings=32):
    """
    return:
        model,input_size
    """
    if modelname in timm.list_models() or modelname.split('.')[0] in timm.list_models():
        assert modelname in pretrained_models
        pretrained_cfg = timm.models.create_model(modelname).default_cfg
        pretrained_cfg['file']=pretrained_paths[modelname]
        model = timm.create_model(model_name=modelname,pretrained=True,num_classes=num_out_classes,features_only=(not use_fc),in_chans=spa_inc,pretrained_cfg=pretrained_cfg)
        config = resolve_data_config({}, model=model)
        input_size=config['input_size']
        return model,input_size
    elif modelname.lower() == 'eff_cmia':
        model = EFF_CMIA(eff,embddings=embeddings,freq_srm=freq_srm,fusion=fusion)
        return model,global_input_sizes[eff]
    else:
        raise NameError("invalid model name",modelname)
