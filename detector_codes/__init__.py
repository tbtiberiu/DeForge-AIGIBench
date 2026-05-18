import argparse
import importlib
import os
import sys
import urllib.request

import torch
import torch.nn as nn
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from torchvision import transforms
from transformers import CLIPModel

load_dotenv()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def download_weights():
    weights_path = './DeForge-AIGIBench-Models'
    token = os.getenv('HF_TOKEN')
    # Check if directory exists and has more than just the .git folder or similar
    if not os.path.exists(weights_path) or len(os.listdir(weights_path)) <= 1:
        print('Downloading weights from Hugging Face Hub...')
        snapshot_download(
            repo_id='TheKernel01/DeForge-AIGIBench-Models',
            local_dir=weights_path,
            repo_type='model',
            token=token,
        )
    return weights_path


# Download weights on import
download_weights()


class DetectorWrapper:
    def __init__(self):
        self.model = None
        self.transform = None
        self.use_optimal_threshold = False

    @torch.inference_mode()
    def detect(self, data):
        # Default: sigmoid probability where high = fake if output dim is 1
        # If output dim is 2, use softmax[:, 1]
        out = self.model(data)
        if out.shape[1] == 1:
            return out.sigmoid().flatten()
        else:
            return out.softmax(dim=1)[:, 1].flatten()

    def _setup_path(self, path):
        """Append path to sys.path and clear related cached modules to avoid collisions."""
        # Convert relative path to absolute to ensure consistency
        # Assuming we are in detector_codes/ or the root
        if not os.path.isabs(path):
            # Try to find the path relative to the root if not found
            if not os.path.exists(path):
                # If we are in detector_codes, go up one level
                possible_path = os.path.join('..', path)
                if os.path.exists(possible_path):
                    path = possible_path
                else:
                    # Fallback to absolute if possible or just use as is
                    pass

        abs_path = os.path.abspath(path)
        if abs_path not in sys.path:
            sys.path.insert(0, abs_path)

        # Clear modules that might conflict
        conflicting = ('networks', 'models', 'utils', 'data', 'model', 'util')
        to_delete = [
            m
            for m in list(sys.modules.keys())
            if m in conflicting or any(m.startswith(c + '.') for c in conflicting)
        ]
        for m in to_delete:
            del sys.modules[m]


class AIDE_Detector(DetectorWrapper):
    def __init__(self, model_path):
        super().__init__()
        self._setup_path('detector_codes/AIDE-main')
        from data.dct import DCT_base_Rec_Module
        from models.AIDE import AIDE

        self.model = AIDE(resnet_path=None, convnext_path=None)
        self.dct = DCT_base_Rec_Module()
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        msg = self.model.load_state_dict(
            state_dict['model'] if 'model' in state_dict else state_dict, strict=False
        )
        self.model.to(DEVICE).eval()
        self.dct.to(DEVICE)
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.resize = transforms.Resize((256, 256))

    @torch.inference_mode()
    def detect(self, data):
        batch_stacked = []
        for i in range(data.shape[0]):
            img = data[i]
            x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(img)
            stacked = torch.stack(
                [
                    self.normalize(self.resize(x_minmin)),
                    self.normalize(self.resize(x_maxmax)),
                    self.normalize(self.resize(x_minmin1)),
                    self.normalize(self.resize(x_maxmax1)),
                    self.normalize(img),
                ],
                dim=0,
            )
            batch_stacked.append(stacked)
        batch_data = torch.stack(batch_stacked, dim=0)
        out = self.model(batch_data)
        return out.softmax(dim=1)[:, 1].flatten()


class C2P_CLIP_Original(nn.Module):
    def __init__(
        self,
        name='openai/clip-vit-large-patch14',
        num_classes=1,
        hf_token=None,
    ):
        super(C2P_CLIP_Original, self).__init__()
        self.model = CLIPModel.from_pretrained(name, token=hf_token)
        del self.model.text_model
        del self.model.text_projection
        del self.model.logit_scale
        self.model.vision_model.requires_grad_(False)
        self.model.visual_projection.requires_grad_(False)
        self.model.fc = nn.Linear(768, num_classes)
        torch.nn.init.normal_(self.model.fc.weight.data, 0.0, 0.02)

    def encode_image(self, img):
        vision_outputs = self.model.vision_model(
            pixel_values=img,
            output_attentions=self.model.config.output_attentions,
            output_hidden_states=self.model.config.output_hidden_states,
            return_dict=self.model.config.return_dict,
        )
        pooled_output = vision_outputs[1]
        image_features = self.model.visual_projection(pooled_output)
        return image_features

    def forward(self, img):
        image_embeds = self.encode_image(img)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        return self.model.fc(image_embeds)


class C2P_CLIP_Original_Detector(DetectorWrapper):
    def __init__(self, model_path=None):
        super().__init__()
        if model_path is None:
            model_path = 'https://www.now61.com/f/95OefW/C2P_CLIP_release_20240901.zip'
        if model_path.startswith('http'):
            state_dict = torch.hub.load_state_dict_from_url(
                model_path, map_location='cpu', progress=True
            )
        else:
            state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        self.model = C2P_CLIP_Original(
            name='openai/clip-vit-large-patch14',
            num_classes=1,
            hf_token=os.getenv('HF_TOKEN'),
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(DEVICE).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )


class C2P_CLIP_Detector(DetectorWrapper):
    def __init__(self, model_path):
        super().__init__()
        self._setup_path('detector_codes/C2P-CLIP-DeepfakeDetection-main')
        from networks.c2p_clip import C2P_CLIP_Model

        self.model = C2P_CLIP_Model(
            name='openai/clip-vit-large-patch14',
            num_classes=1,
            hf_token=os.getenv('HF_TOKEN'),
        )
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(DEVICE).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def detect(self, img):
        return self.model.detect(img)


class C2P_DINOv2_Detector(DetectorWrapper):
    def __init__(self, model_path=None):
        super().__init__()
        self._setup_path('detector_codes/C2P-DINOv2-main')
        from model import C2P_DINOv2_Model

        self.model = C2P_DINOv2_Model(hf_token=os.getenv('HF_TOKEN'))
        if model_path is not None:
            state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(
                state_dict['model_state_dict']
                if 'model_state_dict' in state_dict
                else state_dict,
                strict=False,
            )
        self.model.to(DEVICE).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def detect(self, img):
        return self.model.detect(img)


class CLIPDetection_Detector(DetectorWrapper):
    def __init__(self, model_path):
        super().__init__()
        self._setup_path('detector_codes/CLIPDetection-main')
        from models.clip_models import CLIPModel

        self.model = CLIPModel(name='ViT-L/14', num_classes=1)
        self.model.load_state_dict(
            torch.load(model_path, map_location='cpu', weights_only=False)
        )
        self.model.to(DEVICE).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )


class CNNDetection_Detector(DetectorWrapper):
    def __init__(self, model_path):
        super().__init__()
        self._setup_path('detector_codes/CNNDetection-master')
        from networks.resnet import resnet50

        self.model = resnet50(num_classes=1)
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(
            state_dict['model'] if 'model' in state_dict else state_dict
        )
        self.model.to(DEVICE).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class DeForge_AI_Detector(DetectorWrapper):
    def __init__(self, model_path=None):
        super().__init__()
        self._setup_path('detector_codes/DeForge-AI-main')
        from model import DeForge_AI_Model

        checkpoint = None
        checkpoint_args = {}
        if model_path is not None:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict):
                checkpoint_args = checkpoint.get('args', {}) or {}
        model_kwargs = {
            'lora_r': checkpoint_args.get('lora_r', 16),
            'lora_alpha': checkpoint_args.get('lora_alpha', 32),
            'lora_dropout': checkpoint_args.get('lora_dropout', 0.5),
            'unfreeze_last_blocks': checkpoint_args.get('unfreeze_last_blocks', 0),
            'image_size': checkpoint_args.get('image_size', 256),
            'forensic_dim': checkpoint_args.get('forensic_dim', 256),
            'hf_token': os.getenv('HF_TOKEN'),
        }
        lora_target_modules = checkpoint_args.get('lora_target_modules')
        if isinstance(lora_target_modules, str):
            model_kwargs['lora_target_modules'] = [
                m.strip() for m in lora_target_modules.split(',') if m.strip()
            ]
        elif lora_target_modules:
            model_kwargs['lora_target_modules'] = lora_target_modules
        self.model = DeForge_AI_Model(**model_kwargs)
        if checkpoint is not None:
            self.model.load_state_dict(
                checkpoint['model_state_dict']
                if 'model_state_dict' in checkpoint
                else checkpoint,
                strict=False,
            )
        self.model.to(DEVICE).eval()
        size = model_kwargs['image_size']
        resize_size = max(int(round(size * 1.15)), size)
        self.transform = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def detect(self, img):
        return self.model.detect(img)


class DFFreq_Detector(DetectorWrapper):
    def __init__(self, model_path):
        super().__init__()
        self._setup_path('detector_codes/DFFreq-main')
        import networks.resnet as resnet_module

        importlib.reload(resnet_module)
        self.model = resnet_module.resnet50(num_classes=1)
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.to(DEVICE).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class Effort_Detector(DetectorWrapper):
    def __init__(self, model_path):
        super().__init__()
        self._setup_path('detector_codes/Effort-AIGI-Detection')
        from models.clip_models import ClipModel

        opt = argparse.Namespace(use_svd=True)
        self.model = ClipModel(
            name='openai/clip-vit-large-patch14',
            opt=opt,
            num_classes=1,
            hf_token=os.getenv('HF_TOKEN'),
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location='cpu', weights_only=False)
        )
        self.model.to(DEVICE).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )


class FreqNet_Detector(DetectorWrapper):
    def __init__(self, model_path):
        super().__init__()
        self._setup_path('detector_codes/FreqNet-DeepfakeDetection-main')
        from networks.freqnet import FreqNet

        self.model = FreqNet(num_classes=1)
        self.model.load_state_dict(
            torch.load(model_path, map_location='cpu', weights_only=False)
        )
        self.model.to(DEVICE).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class GramNet_Detector(DetectorWrapper):
    def __init__(self, model_path):
        super().__init__()
        self._setup_path('detector_codes/Gram-Net-main')
        import networks.resnet as resnet_module

        importlib.reload(resnet_module)
        self.model = resnet_module.resnet18(num_classes=1)
        self.model.load_state_dict(
            torch.load(model_path, map_location='cpu', weights_only=False)
        )
        self.model.to(DEVICE).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class LGrad_Detector(DetectorWrapper):
    def __init__(self, model_path):
        super().__init__()
        self._setup_path('detector_codes/LGrad-master/CNNDetection')
        import networks.resnet as resnet_module

        importlib.reload(resnet_module)
        self.model = resnet_module.resnet50(num_classes=1)
        self.model.load_state_dict(
            torch.load(model_path, map_location='cpu', weights_only=False)
        )
        self.model.to(DEVICE).eval()
        self._setup_path('detector_codes/LGrad-master/img2gad_pytorch')
        from models import build_model

        self.discriminator = build_model(
            gan_type='stylegan',
            module='discriminator',
            resolution=256,
            label_size=0,
            image_channels=3,
        )
        disc_path = 'DeForge-AIGIBench-Models/LGrad-master/karras2019stylegan-bedrooms-256x256_discriminator.pth'
        if not os.path.exists(disc_path):
            os.makedirs(os.path.dirname(disc_path), exist_ok=True)
            urllib.request.urlretrieve(
                'https://lid-1302259812.cos.ap-nanjing.myqcloud.com/tmp/karras2019stylegan-bedrooms-256x256_discriminator.pth',
                disc_path,
            )
        self.discriminator.load_state_dict(
            torch.load(disc_path, map_location='cpu', weights_only=False), strict=True
        )
        self.discriminator.to(DEVICE).eval()
        self.transform = transforms.Compose(
            [transforms.Resize((256, 256)), transforms.ToTensor()]
        )
        self.transform_disc = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )
        self.transform_resnet = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def detect(self, data):
        disc_input = self.transform_disc(data)
        disc_input.requires_grad = True
        with torch.enable_grad():
            pre = self.discriminator(disc_input)
            grad = torch.autograd.grad(
                pre.sum(),
                disc_input,
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )[0]
        b, c, h, w = grad.shape
        grad_flat = grad.view(b, -1)
        grad_min = grad_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        grad_norm = grad - grad_min
        grad_flat_norm = grad_norm.view(b, -1)
        grad_max = grad_flat_norm.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        grad_norm = torch.where(grad_max != 0, grad_norm / grad_max, grad_norm)
        resnet_input = self.transform_resnet(grad_norm)
        with torch.no_grad():
            out = self.model(resnet_input)
            if out.shape[1] == 1:
                return out.sigmoid().flatten()
            else:
                return out.softmax(dim=1)[:, 1].flatten()


class LaDeDa_Detector(DetectorWrapper):
    def __init__(self, model_path):
        super().__init__()
        self._setup_path(
            'detector_codes/RealTime-DeepfakeDetection-in-the-RealWorld-main'
        )
        from networks.LaDeDa import LaDeDa9

        self.model = LaDeDa9(num_classes=1)
        self.model.fc = torch.nn.Linear(2048, 1)
        from collections import OrderedDict
        from copy import deepcopy

        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        pretrained_dict = OrderedDict()
        for ki in state_dict.keys():
            pretrained_dict[ki] = deepcopy(state_dict[ki])
        self.model.load_state_dict(pretrained_dict, strict=True)
        self.model.to(DEVICE).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class NPR_Detector(DetectorWrapper):
    def __init__(self, model_path):
        super().__init__()
        self._setup_path('detector_codes/NPR-DeepfakeDetection-main')
        from networks.resnet import resnet50

        self.model = resnet50(num_classes=1)
        self.model.load_state_dict(
            torch.load(model_path, map_location='cpu', weights_only=False)
        )
        self.model.to(DEVICE).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class RIGID_Detector(DetectorWrapper):
    def __init__(self, model_path=None):
        super().__init__()
        self.use_optimal_threshold = True
        self._setup_path('detector_codes/RIGID-main')
        from rigid_detector import RIGID_Detector as RIGID_Impl

        self.model = RIGID_Impl(lamb=0.05)
        self.model.model.to(DEVICE)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    @torch.inference_mode()
    def detect(self, data):
        return self.model.detect(data)


class Resnet50_Detector(DetectorWrapper):
    def __init__(self, model_path):
        super().__init__()
        self._setup_path('detector_codes/Resnet50-main')
        from networks.resnet import resnet50

        self.model = resnet50(num_classes=1)
        self.model.load_state_dict(
            torch.load(model_path, map_location='cpu', weights_only=False)
        )
        self.model.to(DEVICE).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class SAFE_Detector(DetectorWrapper):
    def __init__(self, model_path):
        super().__init__()
        self._setup_path('detector_codes/SAFE-main')
        from models.resnet import resnet50

        self.model = resnet50(num_classes=2)
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(
            state_dict['model'] if 'model' in state_dict else state_dict, strict=True
        )
        self.model.to(DEVICE).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


weight_mapping = {
    'AIDE': './DeForge-AIGIBench-Models/AIDE-main/model_epoch_best.pth',
    'C2P-CLIP': './DeForge-AIGIBench-Models/C2P-CLIP-DeepfakeDetection-main/model_epoch_best.pth',
    'C2P-CLIP-Original': None,
    'C2P-DINOv2': './DeForge-AIGIBench-Models/C2P-DINOv2-main/model_epoch_best.pth',
    'CLIPDetection': './DeForge-AIGIBench-Models/CLIPDetection-main/model_epoch_best.pth',
    'CNNDetection': './DeForge-AIGIBench-Models/CNNDetection-master/model_epoch_best.pth',
    'DeForge-AI': './DeForge-AIGIBench-Models/DeForge-AI-main/model_epoch_best.pth',
    'DFFreq': './DeForge-AIGIBench-Models/DFFreq-main/model_epoch_best.pth',
    'Effort': './DeForge-AIGIBench-Models/Effort-AIGI-Detection/model_epoch_best.pth',
    'FreqNet': './DeForge-AIGIBench-Models/FreqNet-DeepfakeDetection-main/model_epoch_best.pth',
    'GramNet': './DeForge-AIGIBench-Models/Gram-Net-main/model_epoch_best.pth',
    'LaDeDa': './DeForge-AIGIBench-Models/RealTime-DeepfakeDetection-in-the-RealWorld-main/model_epoch_best.pth',
    'LGrad': './DeForge-AIGIBench-Models/LGrad-master/model_epoch_best.pth',
    'NPR': './DeForge-AIGIBench-Models/NPR-DeepfakeDetection-main/model_epoch_best.pth',
    'RIGID': None,
    'Resnet50': './DeForge-AIGIBench-Models/Resnet50-main/model_epoch_best.pth',
    'SAFE': './DeForge-AIGIBench-Models/SAFE-main/model_epoch_best.pth',
}

detector_classes = {
    'AIDE': AIDE_Detector,
    'C2P-CLIP': C2P_CLIP_Detector,
    'C2P-CLIP-Original': C2P_CLIP_Original_Detector,
    'C2P-DINOv2': C2P_DINOv2_Detector,
    'CLIPDetection': CLIPDetection_Detector,
    'CNNDetection': CNNDetection_Detector,
    'DeForge-AI': DeForge_AI_Detector,
    'DFFreq': DFFreq_Detector,
    'Effort': Effort_Detector,
    'FreqNet': FreqNet_Detector,
    'GramNet': GramNet_Detector,
    'LaDeDa': LaDeDa_Detector,
    'LGrad': LGrad_Detector,
    'NPR': NPR_Detector,
    'RIGID': RIGID_Detector,
    'Resnet50': Resnet50_Detector,
    'SAFE': SAFE_Detector,
}
