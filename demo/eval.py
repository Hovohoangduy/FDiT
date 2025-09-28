import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import imageio
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os
import timeit
from PIL import Image
from misc import grid2fig, conf2fig
import random
from DM.modules.fdit import FDiT
from misc import resize
import cv2
import math

# metric
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance

start = timeit.default_timer()
root_dir = 'datasets/UTD-MHAD/demo'
GPU = "0"
postfix = "-j-sl-random-of-tr-rmm"
INPUT_SIZE = 128
N_FRAMES = 40
RANDOM_SEED = 2222
MEAN = (0.0, 0.0, 0.0)
cond_scale = 1.
only_use_flow = "onlyflow" in postfix or "-of" in postfix
RESTORE_FROM = "log/mhad128/snaps_diff/flowdiff.pth"
AE_RESTORE_FROM = "log/mhad128/snapshots/RegionMM.pth"

MODEL_DIM = 128
MODEL_DEPTH = 4
MODEL_HEADS = 2
MODEL_DIM_HEAD = 32
MODEL_MLP_DIM = 512
DIFF_TIMESTEPS = 1000
DDIM_ETA = 0.0
ADAM_BETAS = (0.9, 0.999)
config_pth = "config/mhad128.yaml"
CKPT_DIR = os.path.join(root_dir, "demo" + postfix)
os.makedirs(CKPT_DIR, exist_ok=True)
print(root_dir)
print(postfix)
print("RESTORE_FROM:", RESTORE_FROM)
print("AE_RESTORE_FROM:", AE_RESTORE_FROM)
print("cond scale:", cond_scale)


def get_arguments():
    parser = argparse.ArgumentParser(description="Flow Diffusion")
    parser.add_argument("--num-workers", default=1)
    parser.add_argument("--gpu", default=GPU, help="choose gpu device.")
    parser.add_argument("--set-start", default=False)
    parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency')
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED, help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", default=RESTORE_FROM)
    parser.add_argument("--fp16", default=False)
    return parser.parse_args()


args = get_arguments()


def sample_img(rec_img_batch, index):
    rec_img = rec_img_batch[index].permute(1, 2, 0).data.cpu().numpy().copy()
    rec_img += np.array(MEAN) / 255.0
    rec_img[rec_img < 0] = 0
    rec_img[rec_img > 1] = 1
    rec_img *= 255
    return np.array(rec_img, np.uint8)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.enabled = True
    cudnn.benchmark = True
    setup_seed(args.random_seed)

    model = FDiT(
        img_size=INPUT_SIZE // 4,
        num_frames=N_FRAMES,
        sampling_timesteps=DIFF_TIMESTEPS,
        null_cond_prob=0.1,
        ddim_sampling_eta=DDIM_ETA,
        timesteps=DIFF_TIMESTEPS,
        dim=MODEL_DIM,
        depth=MODEL_DEPTH,
        heads=MODEL_HEADS,
        dim_head=MODEL_DIM_HEAD,
        mlp_dim=MODEL_MLP_DIM,
        lr=1e-4,
        adam_betas=ADAM_BETAS,
        is_train=True,
        only_use_flow=only_use_flow,
        use_residual_flow=True,
        pretrained_pth=AE_RESTORE_FROM,
        config_pth=config_pth
    )
    model.cuda()

    if args.restore_from and os.path.isfile(args.restore_from):
        print(f"=> loading checkpoint '{args.restore_from}'")
        checkpoint = torch.load(args.restore_from, map_location="cuda")

        if 'diffusion' in checkpoint:
            state_dict = checkpoint['diffusion']
            model_dict = model.diffusion.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items()
                             if k in model_dict and v.shape == model_dict[k].shape}

            skipped = [k for k in state_dict.keys() if k not in filtered_dict]
            model_dict.update(filtered_dict)
            model.diffusion.load_state_dict(model_dict, strict=False)

            print(f"=> Loaded diffusion weights (matched keys: {len(filtered_dict)})")
            if skipped:
                print("Skipped keys due to shape mismatch:", skipped)
        else:
            print("=> WARNING: 'diffusion' weights not found in checkpoint")
    else:
        print("NO checkpoint found!")
        exit(-1)

    model.eval()

    action_list = ["right arm swipe to the left"]

    ref_img_path = "demo/examples/a1_s1_t1_000.png"
    ref_img_name = os.path.basename(ref_img_path)[:-4]
    ref_img_npy = imageio.v2.imread(ref_img_path)[:, :, :3]
    ref_img_npy = cv2.resize(ref_img_npy, (336, 480), interpolation=cv2.INTER_AREA)
    ref_img_npy = resize(ref_img_npy, 128, interpolation=cv2.INTER_AREA)
    ref_img_npy = np.asarray(ref_img_npy, np.float32)
    ref_img_npy = ref_img_npy - np.array(MEAN)
    ref_img = torch.from_numpy(ref_img_npy / 255.0)
    ref_img = ref_img.permute(2, 0, 1).float()
    ref_imgs = ref_img.unsqueeze(dim=0).cuda()

    nf = 40
    cnt = 0

    fid = FrechetInceptionDistance(feature=128).cuda()
    inception = InceptionScore().cuda()

    for ref_text in action_list:
        model.set_sample_input(sample_img=ref_imgs, sample_text=[ref_text])
        model.sample_one_video(cond_scale=cond_scale)
        save_src_img = sample_img(ref_imgs, 0)  # ref làm real
        new_im_list = []

        for frame_idx in range(nf):
            save_sample_out_img = sample_img(model.sample_out_vid[:, :, frame_idx], 0)
            new_im_list.append(save_sample_out_img)

            # FID
            fake_tensor_fid = torch.from_numpy(save_sample_out_img).permute(2, 0, 1).unsqueeze(0).to(torch.uint8)
            fid.update(fake_tensor_fid.cuda(), real=False)

            real_tensor_fid = torch.from_numpy(save_src_img).permute(2, 0, 1).unsqueeze(0).to(torch.uint8)
            fid.update(real_tensor_fid.cuda(), real=True)

            # IS 
            inception.update(fake_tensor_fid.cuda())

        video_name = "%04d_%s_%.2f.gif" % (cnt, ref_img_name, cond_scale)
        print(video_name)
        imageio.mimsave(os.path.join(CKPT_DIR, video_name), new_im_list)
        cnt += 1

    fid_score = fid.compute().item()
    is_mean, is_std = inception.compute()
    print(f"FID: {fid_score:.4f}")
    print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()