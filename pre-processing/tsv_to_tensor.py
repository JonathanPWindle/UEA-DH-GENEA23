from wsgiref.headers import tspecials
import numpy as np
import glob
import os

processed_dir = 'F:\\GENEA23\\data\\tst\\tst\\main-agent'

files = sorted(glob.glob('{:s}/audio/*.dat'.format(processed_dir)))
for f in files:
    print(f)
    
    p = np.memmap(f, dtype=np.float32, mode='r')
    p = p.reshape(-1, 534)
    tsv_fname = os.path.basename(f)
    t_split = tsv_fname.split('_')
    tsv_fname = []
    for i in range(0,4):
        tsv_fname.append(t_split[i])
    tsv_fname = '_'.join(tsv_fname)
    print(tsv_fname)
    if  str(tsv_fname) == 'trn_2023_v0_181' or tsv_fname == 'trn_2023_v0_139':
        print('continuing', tsv_fname)
        continue
    tsv_fname += f'_{os.path.basename(processed_dir)}'
    print(tsv_fname+'.npy')

    fname = os.path.join(processed_dir, 'fasttext', os.path.basename(f))
    if not os.path.exists(fname):
        tsv = np.load(f'{processed_dir}/tsv_fasttext/' +tsv_fname+'.npy')
        tsv_fasttext = np.zeros((p.shape[0], 300))
        curr_word = 0
        for idx, i in enumerate(np.arange(0,(p.shape[0]/30),1/30)):
            if tsv[curr_word][1] <= i:
                curr_word += 1
            if curr_word >= tsv.shape[0]:
                print('End of speech')
                break
            if idx >= tsv_fasttext.shape[0]:
                print(idx,tsv_fasttext.shape[0] )
            elif tsv[curr_word][0] <= i and tsv[curr_word][1] >i:
                # print(i, 'Using word: ', curr_word)
                tsv_fasttext[idx] = tsv[curr_word][2:]
                continue
            # elif tsv[curr_word][0] >= i:
                # print(i, 'using default', tsv[-1][1])
        print(tsv_fasttext.shape)
        
        print(fname)
        fp = np.memmap(fname, dtype=tsv_fasttext.dtype, mode='w+', shape=tsv_fasttext.shape)
        fp[:] = tsv_fasttext[:]
    