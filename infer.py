import torch
from glob import glob
import os
import numpy as np
import cv2
from NeuFlow.neuflow import NeuFlow
from NeuFlow.backbone_v7 import ConvBlock
from data_utils import flow_viz


def pad_to_multiple(image, multiple=16, mode='edge'):
    h, w = image.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
    return padded, pad_h, pad_w

def remove_padding(flow, pad_h, pad_w):
    if pad_h == 0 and pad_w == 0:
        return flow
    h, w = flow.shape[:2]
    return flow[:h - pad_h, :w - pad_w]

def get_cuda_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_width, image_height))

    # Check and apply padding if needed
    padded_image, pad_h, pad_w = pad_to_multiple(image, multiple=16)
    image_tensor = torch.from_numpy(padded_image).permute(2, 0, 1).half()
    return image_tensor[None].cuda(), image, pad_h, pad_w  # also return original (unpadded) image

def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
    fusedconv = (
        torch.nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


if __name__ == '__main__':
    image_path_list = sorted(glob('test_images/*.png'))
    vis_path = 'test_results/'

    device = torch.device('cuda')

    model = NeuFlow().to(device)

    checkpoint = torch.load('neuflow_mixed.pth', map_location='cuda')

    model.load_state_dict(checkpoint['model'], strict=True)

    for m in model.modules():
        if type(m) is ConvBlock:
            m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)  # update conv
            m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)  # update conv
            delattr(m, "norm1")  # remove batchnorm
            delattr(m, "norm2")  # remove batchnorm
            m.forward = m.forward_fuse  # update forward

    model.eval()
    model.half()

    image_height , image_width = cv2.imread(image_path_list[0]).shape[:2]
    # pad height/width if needed for model.init_bhwd
    h_pad = (16 - image_height % 16) % 16
    w_pad = (16 - image_width % 16) % 16
    model.init_bhwd(1, image_height + h_pad, image_width + w_pad, 'cuda')

    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    for image_path_0, image_path_1 in zip(image_path_list[:-1], image_path_list[1:]):

        print(image_path_0)

        image_0_tensor, image_0_np, pad_h, pad_w = get_cuda_image(image_path_0)
        image_1_tensor, image_1_np, _, _ = get_cuda_image(image_path_1)  # assume same padding

        file_name = os.path.basename(image_path_1)

        with torch.no_grad():
            flow = model(image_0_tensor, image_1_tensor)[-1][0]
            flow = flow.permute(1, 2, 0).cpu().numpy()
            flow = remove_padding(flow, pad_h, pad_w)  # crop back if padded
            flow = flow_viz.flow_to_image(flow)

            image_1_np = cv2.resize(image_1_np, (image_width, image_height))
            cv2.imwrite(vis_path + file_name, np.vstack([image_1_np, flow]))
