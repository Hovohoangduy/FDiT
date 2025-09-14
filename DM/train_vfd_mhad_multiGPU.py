# === train_vfd_mhad_multiGPU.py (sau khi chỉnh sửa) ===
import argparse
import torch
from torch.utils import data
import numpy as np
import torch.backends.cudnn as cudnn
import os
import os.path as osp
import timeit
import sys
import random
from misc import Logger
from DM.datasets_mhad import MHAD
from DM.modules.vfdm_gentron_multiGPU import FlowDiffusionGenTron
from torch.optim.lr_scheduler import MultiStepLR

# ===== CONFIG =====
BATCH_SIZE = 4
MAX_EPOCH = 1200
epoch_milestones = [800, 1000]
data_dir = "/kaggle/input/mhad-mini/crop_image_mini"
GPU = "0,1"
postfix = "-joint-steplr-random-onlyflow-train-regionmm"
joint = "joint" in postfix or "-j" in postfix
frame_sampling = "random" if "random" in postfix else "uniform"
only_use_flow = "onlyflow" in postfix or "-of" in postfix
null_cond_prob = 0.1 if joint else 0.0
use_residual_flow = "-rf" in postfix
config_pth = "config/mhad128.yaml"
AE_RESTORE_FROM = "/kaggle/input/checkpoints-mhad-clfdm/RegionMM.pth"
INPUT_SIZE = 128
N_FRAMES = 40
LEARNING_RATE = 2e-4
RANDOM_SEED = 1234
MEAN = (0.0, 0.0, 0.0)
RESTORE_FROM = ""
root_dir = 'log'
SNAPSHOT_DIR = os.path.join(root_dir, 'snapshots'+postfix)

MODEL_DIM = 512
MODEL_DEPTH = 12
MODEL_HEADS = 8
MODEL_DIM_HEAD = 64
MODEL_MLP_DIM = 2048
DIFF_TIMESTEPS = 1000
DDIM_ETA = 0.0
ADAM_BETAS = (0.95, 0.999)

# === Logger ===
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
LOG_PATH = SNAPSHOT_DIR + "/B"+format(BATCH_SIZE, "04d")+"E"+format(MAX_EPOCH, "04d")+".log"
sys.stdout = Logger(LOG_PATH, sys.stdout)

# === Argument Parser ===
def get_arguments():
    parser = argparse.ArgumentParser(description="Flow Diffusion")
    parser.add_argument("--gpu", default=GPU)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--restore-from", default=RESTORE_FROM)
    parser.add_argument("--snapshot-dir", default=SNAPSHOT_DIR)
    return parser.parse_args()

# === Setup seed ===
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# === Main function ===
def main():
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.enabled = True
    cudnn.benchmark = True
    setup_seed(args.random_seed)

    model = FlowDiffusionGenTron(
        img_size=INPUT_SIZE // 4,
        num_frames=N_FRAMES,
        sampling_timesteps=DIFF_TIMESTEPS,
        null_cond_prob=null_cond_prob,
        ddim_sampling_eta=DDIM_ETA,
        timesteps=DIFF_TIMESTEPS,
        dim=MODEL_DIM,
        depth=MODEL_DEPTH,
        heads=MODEL_HEADS,
        dim_head=MODEL_DIM_HEAD,
        mlp_dim=MODEL_MLP_DIM,
        is_train=True,
        only_use_flow=only_use_flow,
        use_residual_flow=use_residual_flow,
        pretrained_pth=AE_RESTORE_FROM,
        config_pth=config_pth
    ).cuda()

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    optimizer_diff = torch.optim.Adam(model.module.diffusion.parameters(), lr=args.learning_rate, betas=ADAM_BETAS)
    scheduler = MultiStepLR(optimizer_diff, epoch_milestones, gamma=0.1)

    if args.restore_from and os.path.isfile(args.restore_from):
        print(f"=> Loading checkpoint '{args.restore_from}'")
        checkpoint = torch.load(args.restore_from)
        model.module.diffusion.load_state_dict(checkpoint['diffusion'], strict=False)
        if 'optimizer_diff' in checkpoint:
            optimizer_diff.load_state_dict(checkpoint['optimizer_diff'])

    trainloader = data.DataLoader(
        MHAD(data_dir=data_dir, image_size=INPUT_SIZE, num_frames=N_FRAMES,
             color_jitter=True, split_train_test=True, sampling=frame_sampling, mean=MEAN),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    for epoch in range(MAX_EPOCH):
        model.train()
        for i, batch in enumerate(trainloader):
            ref_img = batch[0].cuda(non_blocking=True)
            real_vid = batch[1].cuda(non_blocking=True)
            ref_text = batch[2].cuda(non_blocking=True)

            loss = model(ref_img, real_vid, ref_text)

            optimizer_diff.zero_grad()
            loss.backward()
            optimizer_diff.step()

        scheduler.step()

        if (epoch+1) % 50 == 0:
            torch.save({
                'diffusion': model.module.diffusion.state_dict(),
                'optimizer_diff': optimizer_diff.state_dict()
            }, osp.join(args.snapshot_dir, f"model_epoch_{epoch+1}.pth"))

if __name__ == '__main__':
    main()