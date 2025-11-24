# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import json
import random
import os
import numpy as np



def generate(args):

    merge_cls = args['merge_cls']
    # torch.manual_seed(seed)
    # Labels to condition the model with (feel free to change):
    class_labels = eval(merge_cls) # 288,291  291, 292
    weight_strategy = args['weight_strategy']

    # Create sampling noise:
    n = 1
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)
    merge_weights = args['merge_weights']
    if merge_weights is not None:
        merge_weights = torch.tensor(eval(merge_weights), device=device).reshape(len(class_labels),1)
    if merge_weights is None:
        merge_weights = torch.tensor([1./len(class_labels) for i in range(len(class_labels))], device=device).reshape(len(class_labels),1)
    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    model_kwargs = dict(y=y, y_null = y_null, merge_weights = merge_weights, cfg_scale=4.0)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, weight_strategy=weight_strategy, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

   
    img_name = args['name']
    save_image(samples, img_name, nrow=4, normalize=True, value_range=(-1, 1))

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                 
        os.makedirs(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-ep", type=int, default=0)
    parser.add_argument("--end-ep", type=int, default=0)
    args_ = parser.parse_args()
    gen_epochs = [i for i in range(args_.start_ep, args_.end_ep)]


    num_sampling_steps = 50
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_size = 256 // 8
    model = DiT_models['DiT-XL/2'](
        input_size=latent_size,
        num_classes=1000
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = "DiT-XL-2-256x256.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()
    diffusion = create_diffusion(str(num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
    miniid_to_1kid = json.load(open('../data/mini_imagenet/miniid_to_1kid.json', "r"))
    data_dir = '../data/AIGC/'
    img_dir = data_dir+'imgs/'
    label_dir = data_dir+'labels/'
    mkdir(img_dir)
    mkdir(label_dir)
    imgnetid2miniid = {v: k for k, v in miniid_to_1kid.items()}
    ids = [eval(x) for x in miniid_to_1kid.values()]
    alpha = 1.0
    for epoch in gen_epochs:
        label_dict = {}
        epoch_dir = os.path.join(img_dir, 'ep_'+str(epoch))
        mkdir(epoch_dir)
        for each in ids:
            for count in range(480):
                pair_id = random.choice(ids)
                if pair_id == each:
                    merge_cls = [each]
                    merge_weights = [1]
                    a = imgnetid2miniid[str(each)]
                    lst = [a,1]
                else:
                    merge_cls = [each, pair_id]
                    rat_a = np.sqrt(1. - np.random.beta(alpha, alpha)) 
                    rat_b = 1- rat_a
                    merge_weights = [rat_a, rat_b]
                    merge_weights.sort(reverse=True)
                    a = imgnetid2miniid[str(each)]
                    b = imgnetid2miniid[str(pair_id)]
                    lst = [a,b,round(merge_weights[0],2),round(merge_weights[1],2)]
                new_name = str(count) + '_' + '_'.join(map(lambda x:str(x),lst)) + '.png'
                label_dict[new_name] = lst
                args = {
                    'num_sampling_steps' : 50,
                    'merge_cls' : merge_cls.__repr__(), 
                    'weight_strategy' : 0,
                    'merge_weights' : merge_weights.__repr__(),
                    'name' : os.path.join(epoch_dir,new_name)
                }
                generate(args)
        json_str = json.dumps(label_dict, indent=4)
        with open(os.path.join(label_dir,'ep_'+str(epoch)+'.json'), 'w') as json_file:
            json_file.write(json_str)
