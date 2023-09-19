import os
import numpy as np
import glob
import torch
from torch import from_numpy
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

try:
    from pase.models.frontend import wf_builder
except:
    print("Cannot load PASE")

pase = wf_builder('pre-processing/pase_files/PASE+.cfg').eval()
pase.load_pretrained('pre-processing/pase_files/FE_e199.ckpt', load_last=True, verbose=True)
pase.cuda()
audio_transform = pase

# Edit this variable
split = 'train'

processed_dir = f"data/{split}/main-agent"
parsed_dir = f"data/tst/{split}/pase_features"

if os.path.exists(parsed_dir) is False:
    os.mkdir(parsed_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_sound_fs = int((16000/30)) + 1
files = sorted(glob.glob('{:s}/audio/*.dat'.format(processed_dir)))
print(files)
for f in files:
    print("Processing: ", os.path.basename(f))

    if os.path.exists(os.path.join(parsed_dir, os.path.basename(f))):
            print("dataprocessing->Already parsed ", os.path.join(parsed_dir, os.path.basename(f)))
            continue

    audio_mmap = np.memmap(f, dtype=np.float32, mode='r')
    print(audio_mmap)

    audio_mmap = audio_mmap.reshape((-1, n_sound_fs))

    print(audio_mmap.shape)

    pase_features = np.zeros((audio_mmap.shape[0], 768))

    for i, frame in enumerate(audio_mmap):
        audio_torch = from_numpy(frame).to(device)
        feature_vec = audio_transform(audio_torch[None,None,:].permute(1,0,2))
        feature_vec = feature_vec.cpu().detach().numpy().flatten()
        pase_features[i] = feature_vec
    
    print(pase_features.shape)
    
    fp = np.memmap(os.path.join(parsed_dir, os.path.basename(f)), dtype=pase_features.dtype, mode='w+', shape=pase_features.shape)
    fp[:] = pase_features[:]
    print(fp.shape)