from torch import nn
import torch
import torch.nn.functional as F
from torchvision.transforms import Resize
from torchvision import transforms
from modules.util import AntiAliasInterpolation2d, TPS
from torchvision import models
import numpy as np
# from modules.resolution_net import resolution
from models.extractor import VitExtractor
# from util.losses import LossG
import yaml
with open("config/default/config.yaml", "r") as f:
    config = yaml.safe_load(f)
cfg = config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, bg_predictor, dense_motion_network, inpainting_network, train_params, *kwargs):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.inpainting_network = inpainting_network
        self.dense_motion_network = dense_motion_network
        self.extractor = VitExtractor(model_name=cfg['dino_model_name'], device=device)
        imagenet_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        global_resize_transform = Resize(cfg['dino_global_patch_size'], max_size=480)
        self.global_transform = transforms.Compose([global_resize_transform,
                                                    imagenet_norm
                                                    ])
        self.bg_predictor = None
        if bg_predictor:
            self.bg_predictor = bg_predictor
            self.bg_start = train_params['bg_start']

        self.train_params = train_params
        self.scales = train_params['scales']

        self.pyramid = ImagePyramide(self.scales, inpainting_network.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']
        self.dropout_epoch = train_params['dropout_epoch']
        self.dropout_maxp = train_params['dropout_maxp']
        self.dropout_inc_epoch = train_params['dropout_inc_epoch']
        self.dropout_startp =train_params['dropout_startp']
        
        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()
    def calculate_global_ssim_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(inputs, outputs):  # avoid memory limitations
            a = self.global_transform(a)
            b = self.global_transform(b)
            # with torch.no_grad():
            target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(a.unsqueeze(0), layer_num=11)
            keys_ssim = self.extractor.get_keys_self_sim_from_input(b.unsqueeze(0), layer_num=11)
            loss += F.mse_loss(keys_ssim, target_keys_self_sim)
        return loss

    def calculate_crop_cls_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(outputs, inputs):  # avoid memory limitations
            a = self.global_transform(a).unsqueeze(0).to(device)
            b = self.global_transform(b).unsqueeze(0).to(device)
            cls_token = self.extractor.get_feature_from_input(a)[-1][0, 0, :]
            # with torch.no_grad():
            target_cls_token = self.extractor.get_feature_from_input(b)[-1][0, 0, :]
            loss += F.mse_loss(cls_token, target_cls_token)
        return loss

    def calculate_global_id_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(inputs, outputs):
            a = self.global_transform(a)
            b = self.global_transform(b)
            # with torch.no_grad():
            keys_a = self.extractor.get_keys_from_input(a.unsqueeze(0), 11)
            keys_b = self.extractor.get_keys_from_input(b.unsqueeze(0), 11)
            loss += F.mse_loss(keys_a, keys_b)
        return loss

    def forward(self, x, epoch):
        kp_source = self.kp_extractor(x['source'])
        kp_driving = self.kp_extractor(x['driving'])
        kp_middle1 = {}
        kp_middle1['fg_kp'] = 0.5*kp_source['fg_kp'] + 0.5*kp_driving['fg_kp']
        bg_param = None
        bg_param2 = None

        if(epoch>=self.dropout_epoch):
            dropout_flag = False
            dropout_p = 0
        else:
            # dropout_p will linearly increase from dropout_startp to dropout_maxp 
            dropout_flag = True
            dropout_p = min(epoch/self.dropout_inc_epoch * self.dropout_maxp + self.dropout_startp, self.dropout_maxp)
        ################################################################
        dense_motion = self.dense_motion_network(source_image=x['source'], kp_driving=kp_middle1,
                                                    kp_source=kp_source, bg_param = None, 
                                                    dropout_flag = dropout_flag, dropout_p = dropout_p)
        generated = self.inpainting_network(x['source'], dense_motion) 
        generated.update({'kp_source': kp_source, 'kp_middle1': kp_middle1, 'kp_driving': kp_driving})
        #######################################################
        mid_img_stod = generated['prediction']
        #######################################################
        if self.bg_predictor:
            if(epoch>=self.bg_start):
                bg_param = self.bg_predictor(mid_img_stod, x['driving'])
        ###################################################
        dense_motion2 = self.dense_motion_network(source_image=mid_img_stod, kp_driving=kp_driving,
                                                 kp_source=kp_middle1, bg_param=bg_param,
                                                 dropout_flag=dropout_flag, dropout_p=dropout_p)

        generated2 = self.inpainting_network(mid_img_stod, dense_motion2)
        ##################################################
        dense_motion3 = self.dense_motion_network(source_image=x['driving'], kp_driving=kp_middle1,
                                                 kp_source=kp_driving, bg_param=None,
                                                 dropout_flag=dropout_flag, dropout_p=dropout_p)
        generated3 = self.inpainting_network(x['driving'], dense_motion3)
        ###################################################
        mid_img_dtos = generated3['prediction']
        ###################################################
        if self.bg_predictor:
            if(epoch>=self.bg_start):
                bg_param2 = self.bg_predictor(mid_img_dtos, x['source'])
        ###################################################
        dense_motion4 = self.dense_motion_network(source_image=mid_img_dtos, kp_driving=kp_source,
                                                 kp_source=kp_middle1, bg_param=bg_param2,
                                                 dropout_flag=dropout_flag, dropout_p=dropout_p)
        generated4 = self.inpainting_network(mid_img_dtos, dense_motion4)
        ####################################################################
        loss_values = {}

        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated2['prediction'])############
        
        pyramide_real_S = self.pyramid(x['source'])
        pyramide_generated2 = self.pyramid(generated4['prediction'])


        # reconstruction loss
        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
                    
            loss_values['perceptual'] = value_total
            
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated2['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real_S['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
                    
            loss_values['perceptual2'] = value_total
            
        loss_structure = self.calculate_global_ssim_loss(generated2['prediction'], x['driving']) + \
                         self.calculate_global_ssim_loss(generated['prediction'], generated3['prediction']) 
                        #  self.calculate_global_ssim_loss(generated4['prediction'], x['source'])  + \
                         
        loss_values['structure'] = loss_structure * 10
        
        loss_apperance = self.calculate_crop_cls_loss(generated['prediction'], x['source'])  + \
                         self.calculate_crop_cls_loss(generated['prediction'], generated3['prediction']) 
                        #  self.calculate_crop_cls_loss(generated3['prediction'], x['driving']) 
                          
        loss_values['apperance'] = loss_apperance * 10
        
        # equivariance loss
        if self.loss_weights['equivariance_value'] != 0:
            transform_random = TPS(mode = 'random', bs = x['driving'].shape[0], **self.train_params['transform_params'])
            transform_grid = transform_random.transform_frame(x['driving'])
            transformed_frame = F.grid_sample(x['driving'], transform_grid, padding_mode="reflection",align_corners=True)
            transformed_kp = self.kp_extractor(transformed_frame)
        
            generated['transformed_frame'] = transformed_frame##############
            generated['transformed_kp'] = transformed_kp################
        
            warped = transform_random.warp_coordinates(transformed_kp['fg_kp'])
            kp_d = kp_driving['fg_kp']
            value = torch.abs(kp_d - warped).mean()
            loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value
        
        
        # warp loss
        if self.loss_weights['warp_loss'] != 0:
            occlusion_map = generated2['occlusion_map']#######
            encode_map = self.inpainting_network.get_encode(x['driving'], occlusion_map)
            decode_map = generated2['warped_encoder_maps']######
            value = 0
            for i in range(len(encode_map)):
                value += torch.abs(encode_map[i]-decode_map[-i-1]).mean()
        
            loss_values['warp_loss'] = self.loss_weights['warp_loss'] * value
        
        # consistency loss
        if self.loss_weights['warp_loss'] != 0:
            decode_map1 = generated['warped_encoder_maps']
            decode_map2 = generated3['warped_encoder_maps']
            value = 0
            for i in range(len(encode_map)):
                value += torch.abs(decode_map1[i]-decode_map2[i]).mean()
            loss_values['con_loss'] = self.loss_weights['warp_loss'] * value
        
        # bg loss
        if self.bg_predictor and epoch >= self.bg_start and self.loss_weights['bg'] != 0:
            bg_param_reverse = self.bg_predictor(x['driving'], generated['prediction'])
            value = torch.matmul(bg_param, bg_param_reverse)
            eye = torch.eye(3).view(1, 1, 3, 3).type(value.type())
            value = torch.abs(eye - value).mean()
            loss_values['bg'] = self.loss_weights['bg'] * value

        return loss_values, generated, generated2
