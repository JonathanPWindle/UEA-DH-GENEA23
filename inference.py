print('inference')
import pickle
import torch
from utils.PoseUtils import PoseUtils
import os
from utils.trainer_utils import save_checkpoint, load_checkpoint
import yaml
import argparse
from scipy.io.wavfile import write
import glob
import numpy as np
from models.CrossAttentiveTransformerXL import CrossAttentiveTransformerXL


print('inference')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


"""
PARSE ARGUMENTS
"""
parser = argparse.ArgumentParser(description='Process hyper-parameters')
parser.add_argument('--config', type=str, default="configs/genea.yaml")
parser.add_argument('--epoch', type=int, default=8)
parser.add_argument('--mem_len', type=int, default=-1)
args = parser.parse_args()
model_epoch = args.epoch

print(model_epoch)

"""
PARSE PARAMS
"""
config_path = args.config
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    config = dotdict(config)
    if args.mem_len != -1:
        config.mem_len = args.mem_len

MODEL_NAME = config.model_name
SEQ_LENGTH = config.seq_length
device = "cuda" if torch.cuda.is_available() else "cpu"

"""
INSTANTIATE MODEL
"""

pose_size = 501
model = CrossAttentiveTransformerXL(pose_size=pose_size, pose_embedding_size=config.embed_size, tgt_len=SEQ_LENGTH+1, ext_len=0, **config)
save_path = f'checkpoints/{MODEL_NAME}'
out_path = f'results/{MODEL_NAME}'

if model_epoch == -1:
    print("BEST MODEL")
    cpkt = load_checkpoint(f'{save_path}/best_model.ckpt',map_location=device)
elif os.path.exists(f'{save_path}/out_checkpoint_{model_epoch}.ckpt'):
    print(f'{save_path}/out_checkpoint_{model_epoch}.ckpt')
    cpkt = load_checkpoint(f'{save_path}/out_checkpoint_{model_epoch}.ckpt',map_location=device)
elif os.path.exists(f'{save_path}/checkpoint{model_epoch}.ckpt'):
    print(f'{save_path}/checkpoint{model_epoch}.ckpt')
    cpkt = load_checkpoint(f'{save_path}/checkpoint{model_epoch}.ckpt',map_location=device)
elif os.path.exists(f'{save_path}/checkpoint{model_epoch+1}.ckpt'):
    model_epoch += 1
    print(f'{save_path}/checkpoint{model_epoch}.ckpt')
    cpkt = load_checkpoint(f'{save_path}/checkpoint{model_epoch}.ckpt',map_location=device)
elif os.path.exists(f'{out_path}/test/{model_epoch}/all_speakers_540/{model_epoch}_model.ckpt'):
    print(f'{save_path}/checkpoint{model_epoch}.ckpt')
    cpkt = load_checkpoint(f'{out_path}/test/{model_epoch}/all_speakers_540/{model_epoch}_model.ckpt',map_location=device)
else:
    print("No path exists to that checkpoint")
    exit()

model.load_state_dict(cpkt['model'])
epoch = cpkt['epoch']

"""
LOAD DATA
"""
scalar = pickle.load(open('utils/ds_files/genea23_scaler.pkl', 'rb'))
save_path = f'checkpoints/{MODEL_NAME}'
out_path = f'results/{MODEL_NAME}/{epoch}/'
data_path = config.test_data_path


if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(out_path):
    os.makedirs(out_path)


offsets = pickle.load(open('utils/ds_files/genea23_offsets.pkl', 'rb'))

"""
INSTANTIATE UTILS
"""
putils = PoseUtils(scalar=scalar, offsets=torch.Tensor(offsets[str(1).zfill(2)]).clone().to(device), device=device, ds='genea')

print(epoch)

model.to(device)

model.eval()

cpkt = {
        'model': model.state_dict(),
        'epoch': epoch
}

if not os.path.exists(f'{out_path}/{epoch}_model.ckpt'):
    save_checkpoint(cpkt, f'{out_path}/{epoch}_model.ckpt', max_keep=None)

files = sorted(glob.glob('{:s}/main-agent/audio/*.dat'.format(data_path)))
inter_files = sorted(glob.glob('{:s}/interloctr/audio/*.dat'.format(data_path)))

random_noise = None

for f, in_f in zip(files, inter_files): 
    f = os.path.basename(f)
    info = f.replace('.dat', '')
    info_split = info.split('_')
    speaker_txt = info_split[-1]


    main_pase_path = os.path.join(data_path, 'main-agent/pase_features', f)
    main_audio_path = os.path.join(data_path, 'main-agent/audio', f)
    main_text_path = os.path.join(data_path, 'main-agent/fasttext', f)

    in_f = os.path.basename(in_f)
    inter_info = in_f.replace('.dat', '')
    inter_info_split = inter_info.split('_')
    inter_speaker_txt = inter_info_split[-1]
    inter_pase_path = os.path.join(data_path, 'interloctr/pase_features', in_f)
    inter_text_path = os.path.join(data_path, 'interloctr/fasttext', in_f)

    inter_audio_path = os.path.join(data_path, 'interloctr/audio', in_f)


    main_audio_mmap = np.memmap(main_audio_path, dtype=np.float32, mode='r')
    main_pase_mmap = np.memmap(main_pase_path, dtype=np.float64, mode='r')
    main_text_mmap = np.memmap(main_text_path, dtype=np.float64, mode='r')

    inter_audio_mmap = np.memmap(inter_audio_path, dtype=np.float32, mode='r')
    inter_pase_mmap = np.memmap(inter_pase_path, dtype=np.float64, mode='r')
    inter_text_mmap = np.memmap(inter_text_path, dtype=np.float64, mode='r')

    main_audio_mmap = main_audio_mmap.reshape(-1, int((16000/30)) + 1)
    main_pase_mmap = main_pase_mmap.reshape(-1, 256*3)
    main_text_mmap = main_text_mmap.reshape(-1, 300)

    
    inter_audio_mmap = inter_audio_mmap.reshape(-1, int((16000/30)) + 1)
    inter_pase_mmap = inter_pase_mmap.reshape(-1, 256*3)
    inter_text_mmap = inter_text_mmap.reshape(-1, 300)

    
    audio_torch = torch.from_numpy(main_audio_mmap.copy()).float().to(device)
    pase_torch = torch.from_numpy(main_pase_mmap.copy()).float()[None,:].to(device)
    text_torch = torch.from_numpy(main_text_mmap.copy()).float()[None,:].to(device)
    print(f, pase_torch.shape, text_torch.shape, audio_torch.shape)
    inter_audio_torch = torch.from_numpy(inter_audio_mmap.copy()).float().to(device)
    inter_pase_torch = torch.from_numpy(inter_pase_mmap.copy()).float()[None,:].to(device)
    inter_text_torch = torch.from_numpy(inter_text_mmap.copy()).float()[None,:].to(device)


    main_speaker = torch.Tensor([int(speaker_txt)-1]).long().repeat(audio_torch.shape[0])[None, :, None].to(device)
    inter_speaker_torch = torch.Tensor([int(inter_speaker_txt)-1]).long().repeat(inter_audio_torch.shape[0])[None, :, None].to(device)

    main_audio_raw = audio_torch.cpu().detach().numpy()
    inter_audio_raw = inter_audio_torch.cpu().detach().numpy()

    write(f'{out_path}/{f}.wav', 16000, main_audio_raw.flatten())
    print('Predicting...')
    mems = tuple()
    full_seq = None
    for j in range(0, pase_torch.shape[1], SEQ_LENGTH):
        audio = pase_torch[:, j:j+(SEQ_LENGTH), :]
        text = text_torch[:, j:j+(SEQ_LENGTH), :]
        speaker = main_speaker[:, j:j+(SEQ_LENGTH), :]

        inter_audio = inter_pase_torch[:, j:j+(SEQ_LENGTH), :]
        inter_text = inter_text_torch[:, j:j+(SEQ_LENGTH), :]
        inter_speaker = inter_speaker_torch[:, j:j+(SEQ_LENGTH), :]

        sampled, mems = model(audio=audio, text=text, speaker_id=speaker, inter_audio=inter_audio, inter_text=inter_text, inter_speaker_id=inter_speaker, mems=mems)

        if full_seq is None:
            full_seq = sampled
        else:
            full_seq = torch.cat((full_seq, sampled), dim=1)
    
    x_seq_hat = full_seq[0]
    
    x_seq_hat = putils.inverse_transform_pose(x_seq_hat, scale_rots=True)
    x_seq_hat, bvh_combined = putils.test_seq_to_pos(x_seq_hat)

    print(pase_torch.shape[1], x_seq_hat.shape, inter_pase_torch.shape)

    header = None
    with open('utils/ds_files/genea_header.bvh', 'r') as file:
        header = file.read()
    
    header += f'Frames: {pase_torch.shape[1]}\nFrame Time:	0.0333333\n'

    np.savetxt(f'{out_path}/{epoch}_{f}.bvh', bvh_combined, header=header, comments='')