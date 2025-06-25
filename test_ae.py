import os
import time
import argparse
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.autoencoder import *
from evaluation import EMD_CD

def visualize_and_save_pointclouds(sample_pcs, num_show=4, titles=None, figsize=(12, 3), save_path='pointcloud_samples.png'):
    if isinstance(sample_pcs, torch.Tensor):
        sample_pcs = sample_pcs.detach().cpu().numpy()
        
    B = sample_pcs.shape[0]
    num_show = min(B, num_show)

    fig = plt.figure(figsize=figsize)
    for i in range(num_show):
        ax = fig.add_subplot(1, num_show, i + 1, projection='3d')
        pc = sample_pcs[i]
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c='blue')
        ax.set_axis_off()
        ax.set_title(titles[i] if titles else f'Sample {i}')
        ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 点云图已保存到: {os.path.abspath(save_path)}")

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./logs_ae/AE_2025_06_24__15_27_15/ckpt_0.003544_5000.pt')
parser.add_argument('--categories', type=str_list, default=['airplane'])
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./data/ply_data_test0.h5')
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

# Logging
save_dir = os.path.join(args.save_dir, 'AE_Ours_%s_%d' % ('_'.join(args.categories), int(time.time())) )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logger = get_logger('test', save_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

# Checkpoint
ckpt = torch.load(args.ckpt)
seed_all(ckpt['args'].seed)

# Datasets and loaders
logger.info('Loading datasets...')
test_dset = ShapeNetCore(
    path=args.dataset_path,
    cates=args.categories,
    split='test',
    scale_mode=ckpt['args'].scale_mode
)
test_loader = DataLoader(test_dset, batch_size=args.batch_size, num_workers=0)

# Model
logger.info('Loading model...')
model = AutoEncoder(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])

all_ref = []
all_recons = []
for i, batch in enumerate(tqdm(test_loader)):
    ref = batch['pointcloud'].to(args.device)
    shift = batch['shift'].to(args.device)
    scale = batch['scale'].to(args.device)
    model.eval()
    with torch.no_grad():
        code = model.encode(ref)
        recons = model.decode(code, ref.size(1), flexibility=ckpt['args'].flexibility).detach()

    ref = ref * scale + shift
    recons = recons * scale + shift
    
    all_ref.append(ref.detach().cpu())
    all_recons.append(recons.detach().cpu())

all_ref = torch.cat(all_ref, dim=0)
all_recons = torch.cat(all_recons, dim=0)

visualize_and_save_pointclouds(all_recons.to(args.device), num_show=20)

logger.info('Saving point clouds...')
np.save(os.path.join(save_dir, 'ref.npy'), all_ref.numpy())
np.save(os.path.join(save_dir, 'out.npy'), all_recons.numpy())

logger.info('Start computing metrics...')
metrics = EMD_CD(all_recons.to(args.device), all_ref.to(args.device), batch_size=args.batch_size)
mmd, cov, knn1 = metrics['MMD-CD'].item(), metrics['COV-CD'].item(), metrics['1NN-CD'].item()
logger.info('MMD-CD:  %.12f' % mmd)
logger.info('COV-CD: %.12f' % cov)
logger.info('1NN-CD: %.12f' % knn1)
