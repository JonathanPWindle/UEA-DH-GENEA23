import pandas as pd
import os
from bvh2json import get_eulers_from_motion, parse_bvh, get_6DOF_angles
import librosa
import numpy as np
from pickle import load

def parse_to_numpy(data_root, processed_path, overwrite=False, overlap_secs=4):
    """
        Reads in each wav/mocap file. Extracts data and writes to npy.
        Args:
        hparams: parameters specifying data locations etc.
        overwrite: specifies whether to overwrite existing files
    """

    if os.path.exists(processed_path) is False:
        os.mkdir(processed_path)

    df = pd.read_csv(data_root + '/../metadata.csv', header=0)

    files = []
    for i in df.iterrows():
        
        files.append({'fname': i[1][0] + '_' + os.path.basename(data_root), 'fingers': i[1][2], 'id': str(i[1][1]).zfill(2)})
    
    importandslice(data_root, files, processed_path, overwrite=overwrite, overlap_secs=overlap_secs)

def importandslice(data_root, files, processed_path, window_secs=20, overlap_secs=4, overwrite=True):

    offsets = load(open('utils/ds_files/genea23_offsets.pkl', 'rb'))

    for f in files:
        if overwrite is not True and os.path.exists(os.path.join(processed_path, 'dof6', '{:s}.dat'.format(f['fname']))):
            print("dataprocessing->Already parsed ", f['fname'])
            continue

        if os.path.exists(os.path.join(data_root, 'bvh', '{:s}.bvh'.format(f['fname']))) is False:
            print("dataprocessing->BVH File does not exist: ", f['fname'])
            continue

        if os.path.exists(os.path.join(data_root, 'wav', '{:s}.wav'.format(f['fname']))) is False:
            print("dataprocessing->WAV File does not exist: ", f['fname'])
            continue

        fps = 30

        a, fs = load_audio_data_librosa(data_root, f)

        a = slice_data(a, window_secs, overlap_secs, fps=fs, audio=True).squeeze()

        m, p = load_bvh_data(data_root, f, offsets=offsets)
        
        m = slice_data(m, window_secs, overlap_secs, fps=fps).squeeze()
        p = slice_data(p, window_secs, overlap_secs, fps=fps).squeeze()

        if a.shape[0] > m.shape[0]:
            a = np.delete(a, -1, axis=0)
        elif a.shape[0] < m.shape[0]:
            m = np.delete(m, -1, axis=0)
            p = np.delete(p, -1, axis=0)

        save_data(processed_path, f, a, m, p)


def save_data(processed_dir, f, a, m, p):
    F_FORMAT = '{:s}_{:s}_{:s}.dat'
    m_out = os.path.join(processed_dir, 'dof6', F_FORMAT.format(f['fname'], f['fingers'], f['id']))
    p_out = os.path.join(processed_dir, 'pos', F_FORMAT.format(f['fname'], f['fingers'], f['id']))
    a_out = os.path.join(processed_dir, 'audio', F_FORMAT.format(f['fname'], f['fingers'], f['id']))

    if os.path.exists(processed_dir + '/dof6') is False:
        os.mkdir(processed_dir + '/dof6')
    if os.path.exists(processed_dir + '/pos') is False:
        os.mkdir(processed_dir + '/pos')
    if os.path.exists(processed_dir + '/audio') is False:
        os.mkdir(processed_dir + '/audio')

    print('dataprocessing->save_data:', m_out)
    generate_memmap(a_out, a)
    generate_memmap(m_out, m)
    generate_memmap(p_out, p)

def generate_memmap(filepath, numpy_array):
    fp = np.memmap(filepath, dtype=numpy_array.dtype, mode='w+', shape=numpy_array.shape)
    fp[:] = numpy_array[:]

def load_bvh_data(data_root, fileinfo, offsets=None):
    file_path_id1 = os.path.join(data_root, 'bvh', '{:s}.bvh'.format(fileinfo['fname']))
    
    
    j, p, m = parse_bvh(file_path_id1,remove_mean=False, offsets=offsets[fileinfo['id']])
    root = j['skeleton']

    dof6 = get_6DOF_angles(m, root)

    dof6 = dof6.reshape((dof6.shape[0], -1))
    m = np.concatenate((m[:,:3], dof6), axis=-1)
    p = p.reshape((p.shape[0], -1))
    

    return m, p

def slice_data(data, window_secs, overlap_secs, fps, audio=False):
    if len(data.shape) == 1:
        data = data[:,None]
    
    window_size = fps*window_secs
    overlap_frames = fps*overlap_secs

    if audio:
        overlap_frames += 1
        window_size += 1

    nframes = data.shape[0]
    n_sequences = int((nframes-overlap_frames)//(window_size-overlap_frames))

    window_size = int(window_size)
    overlap_frames = int(overlap_frames)

    if n_sequences>0:
        sliced = np.zeros((n_sequences, window_size, data.shape[1])).astype(np.float32)
        inc = 0

        for i in range(0,n_sequences):
            if i%3== 0 and  i != 0 and audio:
                inc += 1
            frame_idx = (window_size-overlap_frames) * i + inc

            sliced[i,:,:] = data[frame_idx:frame_idx+window_size,:].copy()
    
    else:
        print("WARNING: data too small for window")
        sliced = np.zeros((0, window_size, data.shape[1])).astype(np.float32)
    
    return sliced


def load_audio_data_librosa(data_root, fileinfo):
    """
    Reads in wav file from twh directory structure
    Args:
    data_root: root of data
    fileinfo: information on clip to be loaded (session)
    """

    file_path_id1 = os.path.join(data_root, 'wav', '{:s}.wav'.format(fileinfo['fname']))

    print('dataprocessing->load_audio_data:', file_path_id1)
    X1, fs1 = librosa.load(file_path_id1, sr=16000)


    return X1, fs1


if __name__ == "__main__":
    # Edit this value
    data_root = 'genea_data/train'
    split = os.path.basename(os.path.normpath(data_root))
    for sp in ['main-agent', 'interloctr']:
        data_root = f"{data_root}/{sp}"
        processed_path = f'data/{split}'
        print(processed_path)

        parse_to_numpy(data_root, processed_path, overlap_secs=0)
