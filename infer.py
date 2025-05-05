import torch
from glob import glob
import os
import numpy as np
import cv2
from NeuFlow.neuflow import NeuFlow
from NeuFlow.backbone_v7 import ConvBlock
from data_utils import flow_viz
from pathlib import Path


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

def get_cuda_image(image_path, image_width, image_height):
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

def compute_dense_optical_flow_tv_l1(I_prev, I_t):
    prev_gray = cv2.cvtColor(I_prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(I_t, cv2.COLOR_BGR2GRAY)
    optical_flow = cv2.optflow.createOptFlow_DualTVL1()
    flow = optical_flow.calc(prev_gray, curr_gray, None)  # shape: (H, W, 2)
    return flow
                                   # Update with your actual flow viz

def compute_dense_flow_for_sequence(base_path):
    image_path_list = sorted(glob(f'{base_path}/*.jpg'))
    flow_path = f'{base_path}/flow_map/'

    device = torch.device('cuda')
    model = NeuFlow().to(device)
    checkpoint = torch.load('neuflow_mixed.pth', map_location='cuda')
    model.load_state_dict(checkpoint['model'], strict=True)

    for m in model.modules():
        if type(m) is ConvBlock:
            m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)
            m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)
            delattr(m, "norm1")
            delattr(m, "norm2")
            m.forward = m.forward_fuse

    model.eval()
    model.half()

    image_height, image_width = cv2.imread(image_path_list[0]).shape[:2]
    h_pad = (16 - image_height % 16) % 16
    w_pad = (16 - image_width % 16) % 16
    model.init_bhwd(1, image_height + h_pad, image_width + w_pad, 'cuda')

    os.makedirs(flow_path, exist_ok=True)

    for image_path_0, image_path_1 in zip(image_path_list[:-1], image_path_list[1:]):
        image_0_tensor, image_0_np, pad_h, pad_w = get_cuda_image(image_path_0, image_width, image_height)
        image_1_tensor, image_1_np, _, _ = get_cuda_image(image_path_1, image_width, image_height)

        file_name = os.path.basename(image_path_1)
        base_name = os.path.splitext(file_name)[0]

        with torch.no_grad():
            # --- NeuFlow prediction ---
            flow = model(image_0_tensor, image_1_tensor)[-1][0]
            flow = flow.permute(1, 2, 0).cpu().numpy()
            flow = remove_padding(flow, pad_h, pad_w)
            flow = flow.astype(np.float32)
            np.save(os.path.join(flow_path, base_name + '_flow.npy'), flow)
            # print(f"Saved NeuFlow to {base_name}_flow.npy")

            # # --- TV-L1 prediction ---
            # flow_tvl1 = compute_dense_optical_flow_tv_l1(image_0_np, image_1_np)
            # flow_tvl1 = flow_tvl1.astype(np.float32)
            # np.save(os.path.join(flow_path, base_name + '_flow_tvl1.npy'), flow_tvl1)

            # --- Visualization ---
            flow_img_neuflow = flow_viz.flow_to_image(flow)
            # flow_img_tvl1 = flow_viz.flow_to_image(flow_tvl1)

            image_1_vis = cv2.resize(image_1_np, (image_width, image_height))
            comparison_image = np.vstack([
                image_1_vis,
                flow_img_neuflow
                # flow_img_tvl1
            ])
            vis_dir = os.path.join(flow_path, "visualisation")
            os.makedirs(vis_dir, exist_ok=True)
            vis_file = os.path.join(vis_dir, base_name + '.jpg')
            cv2.imwrite(vis_file, comparison_image)
            print(f"Saved NeuFlow to {os.path.join(flow_path, base_name + '_flow.npy')} | Image: {vis_file}")
            
            # print(f"Saved NeuFlow to {base_name}_flow.npy | TV-L1 to {base_name}_flow_tvl1.npy | Image: {vis_file}")


# def main_cloth_blow_flow(cloth_blow_path: str):
#     cloth_blow_dir = Path(cloth_blow_path)

#     # List and sort B folders (e.g., B00 to B21)
#     traj_folders = sorted([f for f in cloth_blow_dir.iterdir() if f.is_dir() and f.name.startswith("B")])

#     for traj in traj_folders:
#         image_dir = traj / "images"
#         if image_dir.exists():
#             print(f"▶️ Processing flow for: {image_dir}")
#             compute_dense_flow_for_sequence(str(image_dir))
#         else:
#             print(f"⛔ No images found in {image_dir}, skipping...")

#     print("✅ Done computing flow for all sub-trajectories.")

# # Example usage
# main_cloth_blow_flow("../deformable_slam/deformation_datasets/cloth_datasets/clothblow_test")

compute_dense_flow_for_sequence("../deformable_slam/stereo_datasets/stereo_surgt/case8/image01")
