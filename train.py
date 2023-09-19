from models.CrossAttentiveTransformerXL import CrossAttentiveTransformerXL
import torch
from torch.optim import Adam, AdamW
from utils.PoseUtils import PoseUtils
import os
from utils.trainer_utils import save_checkpoint, load_checkpoint
import random
import yaml
import argparse
from losses.GeometricKinetic import GeometricKinetic
import pickle
import sys
import numpy as np
from datasets.genea23 import GENEAFlat, GENEAOrderedIterator

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

# Set all seed values for reproducability
set_seed(1)

"""
PARSE ARGUMENTS
"""
parser = argparse.ArgumentParser(description='Process hyper-parameters')
parser.add_argument('--config', type=str, default="configs/cross.yaml")
args = parser.parse_args()

"""
PARSE PARAMS
"""
config_path = args.config
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    hyper_params = {**config}
    config = dotdict(config)

BATCH_SIZE = config.batch_size
EPOCHS = config.epochs
SEQ_LENGTH = config.seq_length
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = config.model_name

"""
LOAD DATA
"""
scalar = pickle.load(open('utils/ds_files/genea23_scaler.pkl', 'rb'))
data = GENEAFlat(config.train_data_path, representation=config.representation, num_frames=config.seq_length, overlap=config.overlap, scalar_type=config.scalar_type, scale=config.scale, scalar=scalar)
val_data = GENEAFlat(config.val_data_path, representation=config.representation, num_frames=config.seq_length, overlap=0, scale=config.scale, scalar=scalar)
train_oi = GENEAOrderedIterator(data.get_data(), BATCH_SIZE, SEQ_LENGTH, device=device, overlap=config.overlap)
val_oi = GENEAOrderedIterator(val_data.get_data(), BATCH_SIZE, SEQ_LENGTH, device=device, overlap=0)

dataloader = train_oi
val_loader = val_oi
save_path = f'checkpoints/{MODEL_NAME}'
out_path = f'results/{MODEL_NAME}'

if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(out_path):
    os.makedirs(out_path)


"""
INSTANTIATE UTILS
"""
pose_size = data[0][1].shape[-1]
putils = PoseUtils(data=data, device=device, ds='genea')

"""
INSTANTIATE MODEL
"""

model = CrossAttentiveTransformerXL(pose_size=pose_size, pose_embedding_size=config.embed_size, tgt_len=SEQ_LENGTH+1, ext_len=0, **config)

model.to(device)
print(model)

"""
OPTIMISER AND LOSS
"""
optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

train_loss = GeometricKinetic(prepend='train', device=device, rotation=True, position=True, velocity=True, **config)
val_loss = GeometricKinetic(prepend='val', device=device, rotation=True, position=True, velocity=True, **config)

"""
TRAINING_LOOP
"""
print('training...')
best_loss = float('inf')
for epoch in range(EPOCHS):
    total_loss = 0
    l_pos_total = 0
    l_vel_total = 0
    l_acc_total = 0
    l_rotation_total = 0
    l_root_total = 0
    l_root_vel_total = 0
    l_kinetic_total = 0
    model.train()
    mems = tuple()
    prev_batch = None
    prev_seq_names = None
    loss_mask = None
    for step, (data_batch, seq_len) in enumerate(dataloader):
        main_audio = data_batch[0].to(device)
        inter_audio = data_batch[6].to(device)
        main_motion = data_batch[1].to(device)
        inter_motion = data_batch[7].to(device)
        main_speaker = data_batch[2].to(device)
        inter_speaker = data_batch[8].to(device)
        main_fingers = data_batch[3].to(device)
        inter_fingers = data_batch[9].to(device)
        main_text = data_batch[4].to(device)
        inter_text = data_batch[10].to(device)
        main_seq_name = data_batch[5]
        inter_seq_name = data_batch[11]

        if prev_seq_names is not None:
            mem_mask = (main_seq_name[:,0,:] == prev_seq_names[:,0,:]).astype(int)
            mask_idx = np.where(mem_mask == 0)[0]
            for layer, m in enumerate(mems):
                m = m.permute(1,0,2)
                m[mask_idx] = torch.zeros_like(m[0])
                m = m.permute(1,0,2)
                mems[layer] = m
            prev_batch[mask_idx] = torch.zeros_like(prev_batch[0])
        
        predicted_poses, mems = model(audio=main_audio, text=main_text, speaker_id=main_speaker, inter_audio=inter_audio, inter_text=inter_text, inter_speaker_id=inter_speaker, mems=mems)

        if prev_batch is not None:
            main_motion = torch.cat([prev_batch[:, -2:, :], main_motion], dim=-2)
            predicted_poses = torch.cat([prev_pred[:, -2:, :], predicted_poses], dim=-2)
        
        poses_pos = putils.inverse_transform_pose(main_motion.detach().clone().reshape(-1,  pose_size), scale_rots=True).reshape(BATCH_SIZE, main_motion.shape[-2],  pose_size)
        gt_root = poses_pos[..., :3]
        poses_pos = putils.seq_to_pos(poses_pos, root_zero=True).to(device)[..., 1:, :]
        pred_pos = putils.inverse_transform_pose(predicted_poses.detach().clone().reshape(-1,  pose_size), scale_rots=True).reshape(BATCH_SIZE, main_motion.shape[-2],  pose_size)
        pred_root = pred_pos[..., :3]
        pred_pos = putils.seq_to_pos(pred_pos, root_zero=True).to(device)[..., 1:, :]

        loss, l_rotation, l_pos, l_vel, l_acc, l_root_vel, l_root, l_kinetic, l_unweighted = train_loss(gt_rot=main_motion, pred_rot=predicted_poses, gt_pos=poses_pos, pred_pos=pred_pos, gt_root=gt_root, pred_root=pred_root)
        if torch.isnan(loss).any():
            sys.exit("NaN in loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        l_rotation_total += l_rotation.item()
        l_pos_total += l_pos.item()
        l_vel_total += l_vel.item()
        l_acc_total += l_acc.item()
        l_root_total += l_root.item()
        l_root_vel_total  += l_root_vel.item()
        l_kinetic_total += l_kinetic.item()

        if prev_batch is not None:
            prev_batch = main_motion[:, -config.mem_len:, :]
            prev_pred = predicted_poses.detach().clone()[:, -config.mem_len:, :]

        else:
            prev_batch = main_motion
            prev_pred = predicted_poses.detach().clone()
    
        prev_seq_names = main_seq_name
    total_loss /= (step + 1)
    l_rotation_total /= (step + 1)
    l_pos_total /= (step + 1)
    l_vel_total /= (step + 1)
    l_acc_total /= (step + 1)
    l_root_total /= (step + 1)
    l_root_vel_total /=  (step + 1)
    l_kinetic_total /=  (step + 1)

    print(f"Epoch {epoch} | Loss: {total_loss} | Simple: {l_rotation_total} | Position: {l_pos_total} | Velocity: {l_vel_total}")

    total_loss_val = 0
    l_rot_total_val = 0
    l_pos_total_val = 0
    l_vel_total_val = 0
    l_acc_total_val = 0
    l_root_total_val = 0
    l_root_vel_total_val = 0
    l_kinetic_total_val = 0

    model.eval()
    mems = tuple()
    prev_batch = None
    prev_seq_names = None
    for step, (data_batch, seq_len) in enumerate(val_loader):
        with torch.no_grad():
            main_audio = data_batch[0].to(device)
            inter_audio = data_batch[6].to(device)
            main_motion = data_batch[1].to(device)
            inter_motion = data_batch[7].to(device)
            main_speaker = data_batch[2].to(device)
            inter_speaker = data_batch[8].to(device)
            main_fingers = data_batch[3].to(device)
            inter_fingers = data_batch[9].to(device)
            main_text = data_batch[4].to(device)
            inter_text = data_batch[10].to(device)
            main_seq_name = data_batch[5]
            inter_seq_name = data_batch[11]

            if prev_seq_names is not None:
                mem_mask = (main_seq_name[:,0,:] == prev_seq_names[:,0,:]).astype(int)
                mask_idx = np.where(mem_mask == 0)[0]
                for layer, m in enumerate(mems):
                    m = m.permute(1,0,2)
                    m[mask_idx] = torch.zeros_like(m[0])
                    m = m.permute(1,0,2)
                    mems[layer] = m
                prev_batch[mask_idx] = torch.zeros_like(prev_batch[0])
            
            predicted_poses, mems = model(audio=main_audio, text=main_text, speaker_id=main_speaker, inter_audio=inter_audio, inter_text=inter_text, inter_speaker_id=inter_speaker, mems=mems)

            if prev_batch is not None:
                main_motion = torch.cat([prev_batch[:, -2:, :], main_motion], dim=-2)
                predicted_poses = torch.cat([prev_pred[:, -2:, :], predicted_poses], dim=-2)
            
            poses_pos = putils.inverse_transform_pose(main_motion.detach().clone().reshape(-1,  pose_size), scale_rots=True).reshape(BATCH_SIZE, main_motion.shape[-2],  pose_size)
            gt_root = poses_pos[..., :3]
            poses_pos = putils.seq_to_pos(poses_pos, root_zero=True).to(device)[..., 1:, :]
            pred_pos = putils.inverse_transform_pose(predicted_poses.detach().clone().reshape(-1,  pose_size), scale_rots=True).reshape(BATCH_SIZE, main_motion.shape[-2],  pose_size)
            pred_root = pred_pos[..., :3]
            pred_pos = putils.seq_to_pos(pred_pos, root_zero=True).to(device)[..., 1:, :]

            loss, l_rotation, l_pos, l_vel, l_acc, l_root_vel, l_root, l_kinetic, l_unweighted = train_loss(gt_rot=main_motion, pred_rot=predicted_poses, gt_pos=poses_pos, pred_pos=pred_pos, gt_root=gt_root, pred_root=pred_root)

            total_loss_val += loss.item()
            l_rot_total_val += l_rotation.item()
            l_pos_total_val += l_pos.item()
            l_vel_total_val += l_vel.item()
            l_acc_total_val += l_acc.item()
            l_root_total_val += l_root.item()
            l_root_vel_total_val += l_root_vel.item()
            l_kinetic_total_val += l_kinetic.item()

            if prev_batch is not None:
                prev_batch = main_motion[:, -config.mem_len:, :]
                prev_pred = predicted_poses.detach().clone()[:, -config.mem_len:, :]

            else:
                prev_batch = main_motion
                prev_pred = predicted_poses.detach().clone()
        
            prev_seq_names = main_seq_name
    total_loss_val /= (step + 1)
    l_rot_total_val /= (step + 1)
    l_pos_total_val /= (step + 1)
    l_vel_total_val /= (step + 1)
    l_acc_total_val /= (step + 1)
    l_root_total_val /= (step + 1)
    l_root_vel_total_val /= (step + 1)
    l_kinetic_total_val /= (step + 1)

    print(f"Epoch {epoch} | Loss: {total_loss_val} | Simple: {l_rot_total_val} | Position: {l_pos_total_val} | Velocity: {l_vel_total_val}")
    
    cpkt = {
            'model': model.state_dict(),
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict()
    }

    fname = os.path.join(save_path, 'checkpoint'+str(epoch)+'.ckpt')

    if total_loss_val < best_loss:
        save_checkpoint(cpkt, fname, max_keep=1,  is_best=True)
        best_loss = total_loss_val
    else:
        save_checkpoint(cpkt, fname, max_keep=1)

    save_jump = 50
    if epoch%save_jump == 0 or epoch == 2:
        model_path = os.path.join(save_path, 'checkpoint'+str(epoch)+'.ckpt')
        cpkt = load_checkpoint(model_path,map_location=device)
        fname = os.path.join(save_path, 'out_checkpoint_'+str(epoch)+'.ckpt')
        torch.save(cpkt, fname)