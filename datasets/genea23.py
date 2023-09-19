from torch.utils.data import Dataset
import warnings
import torch
import numpy as np
import os
import glob
from pickle import load
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sys
sys.path.append('utils')
from utils.PoseUtils import PoseUtils
warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.manual_seed(0)
np.random.seed(0)

class GENEAFlat(Dataset):
    def __init__(self, processed_dir, num_frames=30, overlap=0, audio_type='pase', device='cpu',scalar=None, scalar_type='standard', scale=True, **kwargs):
        self.num_frames = num_frames
        self.n_sound_fs = int((16000/30))
        self.processed_dir = processed_dir
        self.overlap = overlap
        self.audio_type = audio_type
        self.device = device
        self.data_store = {}

        self.offsets = load(open('utils/ds_files/genea23_offsets.pkl', 'rb'))
        self.all_motion = None
        self.all_audio = None
        self.all_raw_audio = None
        self.all_positions = None
        self.all_speaker = None
        self.all_has_fingers = None
        self.all_offsets = None
        self.interloctr_all_motion = None
        self.interloctr_all_audio = None
        self.interloctr_all_raw_audio = None
        self.interloctr_all_positions = None
        self.interloctr_all_speaker = None
        self.interloctr_all_has_fingers = None
        self.interloctr_all_offsets = None


        if self.audio_type == 'pase':
            self.n_sound_fs = 3 * 256
        else:
            self.n_sound_fs += 1

        self.audio_dtype = np.float32
        if self.audio_type == 'pase':
            self.audio_dtype = np.float64

        print('GENEADatset->__init__: Loading data information')
        if os.path.exists(processed_dir) is False:
            print('GENEADataset->__init__: ERROR! Cannot find processed directory')

        self.data = []

        main_files = sorted(glob.glob('{:s}/main-agent/audio/*.dat'.format(processed_dir)))
        inter_files = sorted(glob.glob('{:s}/interloctr/audio/*.dat'.format(processed_dir)))
        for f, in_f in zip(main_files, inter_files):
            if 'trn_2023_v0_139' not in f and 'trn_2023_v0_181' not in f:
                f = os.path.basename(f)
                info = f.replace('.dat', '')
                info_split = info.split('_')
                speaker = info_split[-1]
                fingers = True if info_split[-2] == 'incl' else False
                
                in_f = os.path.basename(in_f)
                interloctr_info = in_f.replace('.dat', '')
                interloctr_info_split = interloctr_info.split('_')
                interloctr_speaker = interloctr_info_split[-1]
                interloctr_fingers = True if interloctr_info_split[-2] == 'incl' else False

                if self.audio_type == 'pase':
                    audio_path = os.path.join(self.processed_dir, 'main-agent/pase_features', f)
                    interloctr_audio_path = os.path.join(self.processed_dir, 'interloctr/pase_features', in_f)

                    self.audio_dtype = np.float64
                else:
                    audio_path = os.path.join(self.processed_dir, 'main-agent/audio', f)

                motion_path = os.path.join(self.processed_dir, 'main-agent/dof6', f)
                interloctr_motion_path = os.path.join(self.processed_dir, 'interloctr/dof6', in_f)

                pos_path = os.path.join(self.processed_dir, 'main-agent/pos', f)
                interloctr_pos_path = os.path.join(self.processed_dir, 'interloctr/pos', in_f)

                raw_audio_path = os.path.join(self.processed_dir, 'main-agent/audio', f)
                interloctr_raw_audio_path = os.path.join(self.processed_dir, 'interloctr/audio', in_f)

                text_path = os.path.join(self.processed_dir, 'main-agent/fasttext', f)
                interloctr_text_path = os.path.join(self.processed_dir, 'interloctr/fasttext', in_f)


                audio_mmap = np.memmap(audio_path, dtype=self.audio_dtype, mode='r')
                interloctr_audio_mmap = np.memmap(interloctr_audio_path, dtype=self.audio_dtype, mode='r')
                
                motion_mmap = np.memmap(motion_path, dtype=np.float32, mode='r')
                interloctr_motion_mmap = np.memmap(interloctr_motion_path, dtype=np.float32, mode='r')
                
                motion_mmap = motion_mmap.reshape((-1, 501))
                interloctr_motion_mmap = interloctr_motion_mmap.reshape((-1, 501))

                motion_samples = motion_mmap.shape[0]
                interloctr_motion_samples = interloctr_motion_mmap.shape[0]

                audio_samples = audio_mmap.shape[0]/self.n_sound_fs
                interloctr_audio_samples = interloctr_audio_mmap.shape[0]/self.n_sound_fs


                

                if audio_samples == motion_samples == interloctr_audio_samples == interloctr_motion_samples:
                    audio_mmap = np.memmap(audio_path, dtype=self.audio_dtype, mode='r')
                    interloctr_audio_mmap = np.memmap(interloctr_audio_path, dtype=self.audio_dtype, mode='r')

                    audio_mmap = audio_mmap.reshape(-1, self.n_sound_fs)
                    interloctr_audio_mmap = interloctr_audio_mmap.reshape(-1, self.n_sound_fs)

                    raw_audio_mmap = np.memmap(raw_audio_path, dtype=np.float32, mode='r')
                    interloctr_raw_audio_mmap = np.memmap(interloctr_raw_audio_path, dtype=np.float32, mode='r')

                    raw_audio_mmap  = raw_audio_mmap.reshape(-1, int(16000/30)+1)
                    interloctr_raw_audio_mmap  = interloctr_raw_audio_mmap.reshape(-1, int(16000/30)+1)


                    motion_mmap = motion_mmap.reshape(-1, 501)
                    interloctr_motion_mmap = interloctr_motion_mmap.reshape(-1, 501)

                    text_mmap = np.memmap(text_path, dtype=np.float64, mode='r')
                    interloctr_text_mmap = np.memmap(interloctr_text_path, dtype=np.float64, mode='r')

                    text_mmap = text_mmap.reshape(-1, 300)
                    interloctr_text_mmap = interloctr_text_mmap.reshape(-1, 300)

                    pos_mmap = np.memmap(pos_path, dtype=np.float32)
                    interloctr_pos_mmap = np.memmap(interloctr_pos_path, dtype=np.float32)

                    pos_mmap = pos_mmap.reshape(-1, 330)
                    interloctr_pos_mmap = interloctr_pos_mmap.reshape(-1, 330)

                    audio_torch = torch.from_numpy(audio_mmap.copy()).float()
                    interloctr_audio_torch = torch.from_numpy(interloctr_audio_mmap.copy()).float()
                   
                    motion_torch = torch.from_numpy(motion_mmap.copy())
                    interloctr_motion_torch = torch.from_numpy(interloctr_motion_mmap.copy())

                    position_torch = torch.from_numpy(pos_mmap.copy())
                    interloctr_position_torch = torch.from_numpy(interloctr_pos_mmap.copy())

                    raw_audio_torch = torch.from_numpy(raw_audio_mmap.copy())
                    interloctr_raw_audio_torch = torch.from_numpy(interloctr_raw_audio_mmap.copy())

                    text_torch = torch.from_numpy(text_mmap.copy()).float()
                    interloctr_text_torch = torch.from_numpy(interloctr_text_mmap.copy()).float()

                    sample_range = int((motion_torch.shape[0])/self.num_frames) * self.num_frames
                    motion_torch = motion_torch[180:sample_range, :]
                    interloctr_motion_torch = interloctr_motion_torch[180:sample_range, :]

                    audio_torch = audio_torch[180:sample_range, :]
                    interloctr_audio_torch = interloctr_audio_torch[180:sample_range, :]

                    position_torch = position_torch[180:sample_range, :]
                    interloctr_position_torch = interloctr_position_torch[180:sample_range, :]

                    raw_audio_torch = raw_audio_torch[180:sample_range, :]
                    interloctr_raw_audio_torch = interloctr_raw_audio_torch[180:sample_range, :]

                    text_torch = text_torch[180:sample_range, :]
                    interloctr_text_torch = interloctr_text_torch[180:sample_range, :]

                    if self.all_motion is None:
                        self.all_motion = motion_torch
                        self.all_audio = audio_torch
                        self.all_speaker = torch.tensor(int(speaker) - 1).repeat(motion_torch.shape[0])[:,None]
                        self.all_has_fingers = torch.tensor(fingers).repeat(motion_torch.shape[0])[:,None]
                        self.all_text = text_torch
                        self.all_sequence_name_torch = np.expand_dims(np.array([info] * motion_torch.shape[0]), axis=-1)
                        self.interloctr_all_motion = interloctr_motion_torch
                        self.interloctr_all_audio = interloctr_audio_torch
                        self.interloctr_all_speaker = torch.tensor(int(interloctr_speaker) - 1).repeat(interloctr_motion_torch.shape[0])[:,None]
                        self.interloctr_all_has_fingers = torch.tensor(interloctr_fingers).repeat(interloctr_motion_torch.shape[0])[:,None]
                        self.interloctr_all_text = interloctr_text_torch
                        self.interloctr_all_sequence_name_torch = np.expand_dims(np.array([interloctr_info] * interloctr_motion_torch.shape[0]), axis=-1)
                    else:
                        self.all_motion = torch.cat((self.all_motion, motion_torch))
                        self.all_audio = torch.cat([self.all_audio ,audio_torch])
                        self.all_speaker = torch.cat([self.all_speaker ,torch.tensor(int(speaker) - 1).repeat(motion_torch.shape[0])[:,None]])
                        self.all_has_fingers = torch.cat([self.all_has_fingers, torch.tensor(fingers).repeat(motion_torch.shape[0])[:,None]])
                        self.all_text = torch.cat((self.all_text, text_torch))
                        self.all_sequence_name_torch = np.concatenate((self.all_sequence_name_torch, np.expand_dims(np.array([info] * motion_torch.shape[0]), axis=-1)))
                        self.interloctr_all_motion = torch.cat((self.interloctr_all_motion, interloctr_motion_torch))
                        self.interloctr_all_audio = torch.cat([self.interloctr_all_audio ,interloctr_audio_torch])
                        self.interloctr_all_speaker = torch.cat([self.interloctr_all_speaker ,torch.tensor(int(interloctr_speaker) - 1).repeat(interloctr_motion_torch.shape[0])[:,None]])
                        self.interloctr_all_has_fingers = torch.cat([self.interloctr_all_has_fingers, torch.tensor(interloctr_fingers).repeat(interloctr_motion_torch.shape[0])[:,None]])
                        self.interloctr_all_text = torch.cat((self.interloctr_all_text, interloctr_text_torch))
                        self.interloctr_all_sequence_name_torch = np.concatenate((self.interloctr_all_sequence_name_torch, np.expand_dims(np.array([interloctr_info] * interloctr_motion_torch.shape[0]), axis=-1)))
                else:
                    print('Samples missmatch: ', f, audio_samples, motion_samples)

        if scalar is None and scale:
            if scalar_type == 'minmax':
                self.mmscaler = MinMaxScaler((-1, 1))
            else:
                self.mmscaler = StandardScaler()
        else:
            self.mmscaler = scalar
        
        if scale:
            self.all_motion = torch.from_numpy(self.mmscaler.transform(self.all_motion.cpu().detach().numpy()))
            self.interloctr_all_motion = torch.from_numpy(self.mmscaler.transform(self.interloctr_all_motion.cpu().detach().numpy()))
        self.mean_pose = torch.Tensor(self.all_motion.mean(axis=-2)).to(device)[None,:]

    def __len__(self):
        return int(self.all_motion.shape[0]/self.num_frames)

    def __getitem__(self, index):
        return (self.all_audio[index: index + self.num_frames], self.all_motion[index: index + self.num_frames], self.all_text[index:index+self.num_frames], self.all_speaker[index: index+self.num_frames], self.interloctr_all_audio[index: index + self.num_frames], self.interloctr_all_motion[index: index + self.num_frames], self.interloctr_all_text[index:index+self.num_frames], self.interloctr_all_speaker[index: index+self.num_frames])


    def get_data(self):
        return (self.all_audio, self.all_motion, self.all_speaker, self.all_has_fingers, self.all_text, self.all_sequence_name_torch, self.interloctr_all_audio, self.interloctr_all_motion, self.interloctr_all_speaker, self.interloctr_all_has_fingers, self.interloctr_all_text, self.interloctr_all_sequence_name_torch)
    
class GENEAOrderedIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None, overlap=0):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0
        self.overlap = overlap
    

        self.device = device

        self.all_motion = data[1]
        self.all_audio = data[0]
        self.all_speaker = data[2]
        self.all_has_fingers = data[3]
        self.all_text = data[4]
        self.all_sequence_name = data[5]
        self.interloctr_all_motion = data[7]
        self.interloctr_all_audio = data[6]
        self.interloctr_all_speaker = data[8]
        self.interloctr_all_has_fingers = data[9]
        self.interloctr_all_text = data[10]
        self.interloctr_all_sequence_name = data[11]


        # Work out how cleanly we can divide the dataset into bsz parts.
        self.n_step = self.all_motion.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        self.all_motion = self.all_motion.narrow(0, 0, self.n_step * bsz)
        self.all_audio = self.all_audio.narrow(0, 0, self.n_step * bsz)
        self.all_speaker = self.all_speaker.narrow(0, 0, self.n_step * bsz)
        self.all_has_fingers = self.all_has_fingers.narrow(0, 0, self.n_step * bsz)
        self.all_text = self.all_text.narrow(0, 0, self.n_step * bsz)
        self.all_sequence_name = self.all_sequence_name[:self.n_step * bsz]
        self.interloctr_all_motion = self.interloctr_all_motion.narrow(0, 0, self.n_step * bsz)
        self.interloctr_all_audio = self.interloctr_all_audio.narrow(0, 0, self.n_step * bsz)
        self.interloctr_all_speaker = self.interloctr_all_speaker.narrow(0, 0, self.n_step * bsz)
        self.interloctr_all_has_fingers = self.interloctr_all_has_fingers.narrow(0, 0, self.n_step * bsz)
        self.interloctr_all_text = self.interloctr_all_text.narrow(0, 0, self.n_step * bsz)
        self.interloctr_all_sequence_name = self.interloctr_all_sequence_name[:self.n_step * bsz]

        # Evenly divide the data across the bsz batches.
        self.all_motion = self.all_motion.view(bsz, -1, self.all_motion.shape[-1]).permute(1,0,2).contiguous().to(device)
        self.all_audio = self.all_audio.view(bsz, -1, self.all_audio.shape[-1]).permute(1,0,2).contiguous().to(device)
        self.interloctr_all_motion = self.interloctr_all_motion.view(bsz, -1, self.interloctr_all_motion.shape[-1]).permute(1,0,2).contiguous().to(device)
        self.interloctr_all_audio = self.interloctr_all_audio.view(bsz, -1, self.interloctr_all_audio.shape[-1]).permute(1,0,2).contiguous().to(device)

        self.all_speaker = self.all_speaker.view(bsz, -1, self.all_speaker.shape[-1]).permute(1,0,2).contiguous().to(device)
        self.all_has_fingers = self.all_has_fingers.view(bsz, -1, self.all_has_fingers.shape[-1]).permute(1,0,2).contiguous().to(device)
        self.all_text = self.all_text.view(bsz, -1, self.all_text.shape[-1]).permute(1,0,2).contiguous().to(device)
        self.all_sequence_name = self.all_sequence_name.reshape(bsz, -1, self.all_sequence_name.shape[-1]).transpose(1,0,2)

        self.interloctr_all_speaker = self.interloctr_all_speaker.view(bsz, -1, self.interloctr_all_speaker.shape[-1]).permute(1,0,2).contiguous().to(device)
        self.interloctr_all_has_fingers = self.interloctr_all_has_fingers.view(bsz, -1, self.interloctr_all_has_fingers.shape[-1]).permute(1,0,2).contiguous().to(device)
        self.interloctr_all_text = self.interloctr_all_text.view(bsz, -1, self.interloctr_all_text.shape[-1]).permute(1,0,2).contiguous().to(device)
        self.interloctr_all_sequence_name = self.interloctr_all_sequence_name.reshape(bsz, -1, self.interloctr_all_sequence_name.shape[-1]).transpose(1,0,2)

        # Number of mini-batches
        self.n_batch = (self.n_step + self.bptt - 1) // self.bptt

    def get_batch(self, i, bptt=None):
        if bptt is None: bptt = self.bptt
        seq_len = min(bptt, self.all_motion.size(0) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)
        motion = self.all_motion[beg_idx:end_idx].permute(1,0,2)
        audio = self.all_audio[beg_idx:end_idx].permute(1,0,2)
        speaker = self.all_speaker[beg_idx:end_idx].permute(1,0,2)
        has_fingers = self.all_has_fingers[beg_idx:end_idx].permute(1,0,2)
        text = self.all_text[beg_idx:end_idx].permute(1,0,2)
        seq_name = self.all_sequence_name[beg_idx:end_idx].transpose(1,0,2)

        interloctr_motion = self.interloctr_all_motion[beg_idx:end_idx].permute(1,0,2)
        interloctr_audio = self.interloctr_all_audio[beg_idx:end_idx].permute(1,0,2)
        interloctr_speaker = self.interloctr_all_speaker[beg_idx:end_idx].permute(1,0,2)
        interloctr_has_fingers = self.interloctr_all_has_fingers[beg_idx:end_idx].permute(1,0,2)
        interloctr_text = self.interloctr_all_text[beg_idx:end_idx].permute(1,0,2)
        interloctr_seq_name = self.interloctr_all_sequence_name[beg_idx:end_idx].transpose(1,0,2)

        data = (audio, motion, speaker, has_fingers, text, seq_name, interloctr_audio, interloctr_motion, interloctr_speaker, interloctr_has_fingers, interloctr_text, interloctr_seq_name)

        return data, seq_len

    def get_fixlen_iter(self, start=0):
        for i in range(start, self.all_motion.size(0) - self.bptt, self.bptt-self.overlap):
            yield self.get_batch(i)

    def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        max_len = self.bptt + max_deviation * std
        i = start
        while True:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
            bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
            data, target, seq_len = self.get_batch(i, bptt)
            i += seq_len
            yield data, target, seq_len
            if i >= self.data.size(0) - 2:
                break

    def __iter__(self):
        return self.get_fixlen_iter()

if __name__ == "__main__":
    val_data = GENEAFlat('F:\\GENEA23\\data\\processed\\val', representation='6dof', num_frames=30, overlap=0, scale=True)

    data = val_data.get_data()

    print(val_data[0][1].shape)
    print(val_data[0][0].shape)

    print(len(val_data))

    print(val_data[-31][8].shape)
    print(val_data[-31][6].shape)

    


    # oi = GENEAOrderedIterator(data, 32, 270, 'cpu', overlap=0)

    # itr = oi.get_fixlen_iter()

    # # all_motion = None
    # count = 0
    # prev_batch = None
    # mems = torch.zeros((32, 30, 200))
    # for batch, (data, seq_len) in enumerate(itr):
    #     # count += 1
    #     count = 0
    #     # print(data[-1][:, 0, :].shape)
    #     print(data[-1][0][0])
    #     print(data[-1][1][0])
    #     if prev_batch:
    #         print((data[-1][:,0,:] == prev_batch[-1][:,0,:]).astype(int))
    #         mem_mask = (data[8][:,0,:] == prev_batch[8][:,0,:]).astype(int)
    #         print(np.where(mem_mask == 0)[0])
    #         mask_idx = np.where(mem_mask == 0)[0]
    #         mems[mask_idx] = torch.zeros_like(mems[0])
    #     # print(mems)
    #     prev_batch = data
    #     mems = torch.ones(32, 300, 200)
        # for a in data:
        #     t = a.element_size() * a.nelement()
        #     print(a.shape)
        #     count += t
        # print(data[4].std(dim=1), data[4].mean(dim=1))
        # print("Total: ", count)
        # print("*" * 100)
    #     print(data[0].shape)
    #     print(data[1].shape)

    #     if all_motion is None:
    #         all_motion = data[1][0]
    #     else:
    #         all_motion = torch.cat([all_motion, data[1][0]])

    #     if batch == 10:
    #         break
    # print(all_motion.shape)
    # poses_pos = putils.inverse_transform_pose(val_data[0][1].detach().clone().reshape(-1,  501))
    # poses_pos = putils.seq_to_pos(poses_pos).to('cpu').cpu().detach().numpy()

    # putils.pose_anim(poses_pos, f'flat_test.mp4')