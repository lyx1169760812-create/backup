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




def main(args):
    # Setup PyTorch:
    num_sampling_steps = args['num_sampling_steps']
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
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



    for merge_cls in args['merge_cls']:
        for seed in args['seed']:
            torch.manual_seed(seed)
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

            # Save and display images:
            img_name = args['name']
            if img_name == 'sample.png':
                class_labels_ = eval(merge_cls)
                merge_weights_ = args['merge_weights']
                if merge_weights_ is None:
                    merge_weights_ = [1./len(class_labels_) for i in range(len(class_labels_))]
                if weight_strategy == 0:
                    img_name_ = '_' + '_'.join([str(round(x,2)) for x in merge_weights_])
                if weight_strategy == 1:
                    img_name_ = '_dynamic'
                img_name = 'vis/' + '_'.join([str(x) for x in eval(merge_cls)]) + img_name_ + '_seed_' + str(seed) + '_step_' + str(num_sampling_steps) + '.png'
            elif img_name[-1] == '/':
                class_labels_ = eval(merge_cls)
                merge_weights_ = args['merge_weights']
                if merge_weights_ is None:
                    merge_weights_ = [1./len(class_labels_) for i in range(len(class_labels_))]
                if weight_strategy == 0:
                    img_name_ = '_' + '_'.join([str(round(x,2)) for x in merge_weights_])
                if weight_strategy == 1:
                    img_name_ = '_dynamic_'
                img_name = img_name + '_'.join([str(x) for x in eval(merge_cls)]) + img_name_ + '_seed_' + str(seed) + '_step_' + str(num_sampling_steps) + '.png'
                print(img_name)
            save_image(samples, img_name, nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    seeds = [i for i in range(10)]

    args = {
        'seed' : seeds,
        'num_sampling_steps' : 50,
        'merge_cls' : ['[88,957]'], #  345牛294熊271 狼  330兔子334刺猬335 松鼠  340斑马343疣猪346水牛  207金毛犬 387小熊猫 277 狐   296北极熊 250二哈 282猫   9鸵鸟 22鹰 84孔雀   35 龟  48 巨蜥 49 尼罗鳄
        'weight_strategy' : 0,
        'merge_weights' : None,
        'name' : "vis_avg/" # vis_avg  vis_channel
    }
    main(args)

    

    # args = {
    #     'seed' : [14],
    #     'num_sampling_steps' : 500,
    #     'merge_cls' : ['[207,207]'], #  345牛294熊271 狼  330兔子334刺猬335 松鼠  340斑马343疣猪346水牛  207金毛犬 387小熊猫 277 狐   296北极熊 250二哈 282猫   9鸵鸟 22鹰 84孔雀   35 龟  48 巨蜥 49 尼罗鳄
    #     'weight_strategy' : 0,
    #     'merge_weights' : None,
    #     'name' : "207_seed14_step500.png"
    # }
    # main(args)
