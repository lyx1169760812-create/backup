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
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    class_labels = eval(args.merge_cls) # 288,291  291, 292
    weight_strategy = args.weight_strategy

    # Create sampling noise:
    n = args.out_img_num
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)
    merge_weights = args.merge_weights
    if merge_weights is not None:
        merge_weights = torch.tensor(eval(merge_weights), device=device).reshape(len(class_labels),1)
    if merge_weights is None:
        merge_weights = torch.tensor([1./len(class_labels) for i in range(len(class_labels))], device=device).reshape(len(class_labels),1)
    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    model_kwargs = dict(y=y, y_null = y_null, merge_weights = merge_weights, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, weight_strategy=weight_strategy, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    img_name = args.name
    if img_name == 'sample.png':
        class_labels_ = eval(args.merge_cls)
        merge_weights_ = args.merge_weights
        if merge_weights_ is None:
            merge_weights_ = [1./len(class_labels_) for i in range(len(class_labels_))]
        if weight_strategy == 0:
            img_name_ = '_' + '_'.join([str(x) for x in merge_weights_])
        if weight_strategy == 1:
            img_name_ = '_dynamic'
        img_name = 'vis/' + '_'.join([str(x) for x in eval(args.merge_cls)]) + img_name_ + '_seed_' + str(args.seed) + '_step_' + str(args.num_sampling_steps) + '.png'
    elif img_name[-1] == '/':
        class_labels_ = eval(args.merge_cls)
        merge_weights_ = args.merge_weights
        if merge_weights_ is None:
            merge_weights_ = [1./len(class_labels_) for i in range(len(class_labels_))]
        if weight_strategy == 0:
            img_name_ = '_' + '_'.join([str(x) for x in merge_weights_])
        if weight_strategy == 1:
            img_name_ = '_dynamic'
        img_name = img_name + '_'.join([str(x) for x in eval(args.merge_cls)]) + img_name_ + '_seed_' + str(args.seed) + '_step_' + str(args.num_sampling_steps) + '.png'
        print(img_name)
    save_image(samples, img_name, nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--merge-cls", type=str, default=250)
    parser.add_argument("--merge-weights", type=str, default=None)
    parser.add_argument("--out-img-num", type=int, default=1)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=512)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--name", type=str, default="sample.png")
    parser.add_argument("--weight-strategy", type=int, default=1)
    args = parser.parse_args()
    main(args)
