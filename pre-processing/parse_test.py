import librosa
import numpy as np
import glob
import pandas as pd
import os
from scipy.io.wavfile import write
from scipy.stats import sigmaclip
from scipy.interpolate import interp1d

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

processed_dir = 'F:\\GENEA23\\data\\tst\\tst\\main-agent\\'
data_root = 'F:\\GENEA23\\data\\tst\\tst\\main-agent'
files = sorted(glob.glob('{:s}/wav/*.wav'.format(data_root)))

df = pd.read_csv(data_root + '/../metadata.csv', header=0)
# df = pd.read_csv(data_root + '/val_metadata.csv', header=None)
F_FORMAT = '{:s}_{:s}_{:s}.dat'
files = []
for i in df.iterrows():
    # fdetails = i.split(',')
    
    files.append({'fname': i[1][0] + '_' + os.path.basename(data_root), 'fingers': i[1][2], 'id': str(i[1][1]).zfill(2)})

# files = [files[6]]

print(files)

for f in files:

    file_path_id1 = os.path.join(data_root, 'wav', '{:s}.wav'.format(f['fname']))

    X, fs = librosa.load(file_path_id1, sr=16000)

    a = slice_data(X, window_secs=1/30, overlap_secs=0,fps=fs, audio=True).squeeze()
    print(a.shape)


    a_out = os.path.join(processed_dir, 'audio', F_FORMAT.format(f['fname'], f['fingers'], f['id']))
    fp = np.memmap(a_out, dtype=a.dtype, mode='w+', shape=a.shape)
    fp[:] = a[:]

    # m = a.mean()

    # np_array = a
    # nmax = max(np_array)
    # nmin = min(np_array)

    # clipped_segments = []
    # inside_clip = False
    # clip_start = 0
    # clip_end = 0
    # for i, sample in enumerate(np_array):
    #     if (sample <= nmin + 1) or (sample >= nmax - 1):
    #         if not inside_clip:
    #             # declare we are inside clipped segment
    #             inside_clip = True
    #             # this is the first clipped sample
    #             clip_start = i
    #     elif inside_clip:
    #         inside_clip = False # no longer inside clipped segment
    #         clip_end = i-1  # previous sample is end of segment
    #         # save segment as tuple
    #         clipped_segment = (clip_start, clip_end)
    #         # store tuple in list of clipped segments
    #         clipped_segments.append(clipped_segment)

    # new_array = np_array.copy()  # make copy of original np_array
    # for segment in clipped_segments:
    #     start = segment[0]
    #     end = segment[1]
    #     x_true = list(range(start - 5, start)) + \
    #                 list(range(end + 1, end + 6))
    #     y_true = [np_array[i] for i in x_true]
    #         # function to predict missing values    
    #     interpolation_function = interp1d(x_true, y_true, kind='cubic')
    #     # indices to pass through function 
    #     x_axis = list(range(start - 5, end + 6))
    #     # new sample values
    #     y_axis_new = [float(int(i)) for i in      
    #                 interpolation_function(x_axis)]

    #     for i, x in enumerate(x_axis):
    #         if start <= x <= end:
    #             new_array[x] = y_axis_new[i]
    # smallest_denomination = 1600
    # # fr = []
    # mute_starts = []
    # eps_factor = 3
    # ii = np.where((a < eps_factor * np.finfo(np.float32).eps) & (a > eps_factor* -np.finfo(np.float32).eps))[0]
    # print(ii[0:12])

    # if len(ii) > 0:
    #     i1s = []
    #     currentidx = ii[0]
    #     chunk_length = 0
    #     # for i in ii:
    #     #     if i == currentidx+1:
    #     #         chunk_length += 1
    #     #         currentidx = i
    #     #     else:
    #     #         print(currentidx, chunk_length)
    #     #         chunk_length = 0
    #             # if currentidx < chunk_length:
    #             #     for j in range(chunk_length)
    #             # for j in range(chunk_length):

    #     for i in ii:
    #         if i > currentidx + smallest_denomination:
    #             mute_starts.append(i)
    #             currentidx = i
    #     print(mute_starts)
    #     for i in mute_starts:
    #         zero_els = np.count_nonzero((a[i:i+smallest_denomination] <  eps_factor* np.finfo(np.float32).eps) & (a[i:i+smallest_denomination] > eps_factor* -np.finfo(np.float32).eps))
    #         if zero_els == smallest_denomination:
    #             i1s.append(np.arange(i,i+smallest_denomination))

    #     for i in mute_starts:
    #         if i - smallest_denomination-100 > 0 and i + smallest_denomination+100 < a.shape[0]:
    #             print(a[i:i+smallest_denomination].shape, a[i-smallest_denomination: i].shape)
    #             a[i:i+smallest_denomination+100] = a[i-smallest_denomination-100: i]
        
    #     # print(i1s)
    #     # if len(i1s) > 0:
    #     #     # Get small sequences of muted sounds
    #     #     a = np.delete(a, np.array(i1s).flatten())
    #     #     print(i1s)
    #     # a[ii] = m
    # ii = np.where((a < eps_factor * np.finfo(np.float32).eps) & (a > eps_factor * -np.finfo(np.float32).eps))[0]
    # for i in ii:
    #     # if i > 100:
    #     a[i] = m

    # c, low, upp = sigmaclip(a)

    # print(c.shape, a.shape)


