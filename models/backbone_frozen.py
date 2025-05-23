from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """
    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool, args):
        super().__init__()
        self.args = args
        self.backbone = backbone

        # # Settings for freezing backbone
        # assert 0 <= args.freeze_backbone_at_layer <= 4
        # for name, parameter in backbone.named_parameters(): parameter.requires_grad_(False)  # First freeze all
        # if train_backbone:
        #     if args.freeze_backbone_at_layer == 0:
        #         for name, parameter in backbone.named_parameters():
        #             if 'layer1' in name or 'layer2' in name or 'layer3' in name or 'layer4' in name:
        #                 parameter.requires_grad_(True)
        #     elif args.freeze_backbone_at_layer == 1:
        #         for name, parameter in backbone.named_parameters():
        #             if 'layer2' in name or 'layer3' in name or 'layer4' in name:
        #                 parameter.requires_grad_(True)
        #     elif args.freeze_backbone_at_layer == 2:
        #         for name, parameter in backbone.named_parameters():
        #             if 'layer3' in name or 'layer4' in name:
        #                 parameter.requires_grad_(True)
        #     elif args.freeze_backbone_at_layer == 3:
        #         for name, parameter in backbone.named_parameters():
        #             if 'layer4' in name:
        #                 parameter.requires_grad_(True)
        #     elif args.freeze_backbone_at_layer == 4:
        #         pass
        #     else:
        #         raise RuntimeError

        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def support_encoding_net(self, x, return_interm_layers=False):
        out: Dict[str, NestedTensor] = {}
        m = x.mask
        # x = self.meta_conv(x.tensors)
        x = self.backbone.conv1(x.tensors)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        if return_interm_layers:
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out['0'] = NestedTensor(x, mask)

        x = self.backbone.layer3(x)
        if return_interm_layers:
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out['1'] = NestedTensor(x, mask)

        x = self.backbone.layer4(x)
        if return_interm_layers:
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out['2'] = NestedTensor(x, mask)

        if return_interm_layers:
            return out
        else:
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out['0'] = NestedTensor(x, mask)
            return out

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self,
                 name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 args):
        self.args = args
        dilation = args.dilation
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        for param in backbone.parameters():
            param.requires_grad = False
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded, cannot use res18 & res34."
        super().__init__(backbone, train_backbone, return_interm_layers, args)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos

    def forward_supp_branch(self, tensor_list: NestedTensor, return_interm_layers=False):
        xs = self[0].support_encoding_net(tensor_list, return_interm_layers=return_interm_layers)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = (args.num_feature_levels > 1)
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args)
    for p in backbone.parameters(): p.requires_grad = False
    backbone.eval()
    model = Joiner(backbone, position_embedding)
    return model
