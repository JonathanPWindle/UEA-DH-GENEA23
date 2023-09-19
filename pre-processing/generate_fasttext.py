"""
Text feature extraction script for GENEA Challenge 2022
(word start time, word end time, word) data in TSV is converted to (word start time, word end time, fasttext word vector)
Download pre-trained word vectors first from https://fasttext.cc/docs/en/english-vectors.html and update 'fasttext_path' in the code below
"""
import csv
import glob
import os
import re
from argparse import ArgumentParser
import fasttext
import numpy as np

fasttext_path = 'pre-processing/fasttext_files/crawl-300d-2M-subword.bin'


def extract_text_features(files, dest_dir):
    word_model = fasttext.load_model(fasttext_path)

    for tsv_path in files:
        print(tsv_path)

        feats = []

        with open(tsv_path) as f:
            tsv_file = csv.reader(f, delimiter="\t")
            for line in tsv_file:
                word_start = float(line[0])
                word_end = float(line[1])
                word_tokens = line[2].split()

                for t_i, token in enumerate(word_tokens):
                    token = re.sub(r"([,.!?])", r"", token.strip())  # trim, remove marks
                    if len(token) > 0:
                        new_s_time = word_start + (word_end - word_start) * t_i / len(word_tokens)
                        new_e_time = word_start + (word_end - word_start) * (t_i + 1) / len(word_tokens)
                        word_vec = word_model.get_word_vector(token)
                        feats.append(np.concatenate((np.array([new_s_time, new_e_time]), word_vec)))

        if len(feats) > 0:
            feats = np.vstack(feats)
            print(feats.shape)  # (no. of words, word vec size + 2)

            out_path = os.path.join(dest_dir, os.path.basename(tsv_path).replace('.tsv', '.npy'))
            # with open(out_path, 'wb') as f:
            np.save(out_path, feats)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--tsv_dir', '-orig', required=True,
                        help="Path where tsv files are stored")
    parser.add_argument('--dest_dir', '-dest', required=True,
                        help="Path where extracted text features will be stored")
    parser.add_argument('--tensor_dir', '-tens', required=True,
                        help="Path where extracted tensor fasttext mmaps will be stored")
    params = parser.parse_args()

    tsv_dir = params.tsv_dir
    dest_dir = params.dest_dir

    files = sorted([f for f in glob.iglob(tsv_dir+'/*.tsv')])
    extract_text_features(files, dest_dir)

    processed_dir = params.tensor_dir

    files = sorted(glob.glob('{:s}/audio/*.dat'.format(processed_dir)))
    for f in files:
        
        p = np.memmap(f, dtype=np.float32, mode='r')
        p = p.reshape(-1, 534)
        tsv_fname = os.path.basename(f)
        t_split = tsv_fname.split('_')
        tsv_fname = []
        for i in range(0,4):
            tsv_fname.append(t_split[i])
        tsv_fname = '_'.join(tsv_fname)
        tsv_fname += f'_{os.path.basename(processed_dir)}'

        fname = os.path.join(processed_dir, 'fasttext', os.path.basename(f))
        if not os.path.exists(fname):
            tsv = np.load(f'{dest_dir}' +tsv_fname+'.npy')
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
                    tsv_fasttext[idx] = tsv[curr_word][2:]

            fp = np.memmap(fname, dtype=tsv_fasttext.dtype, mode='w+', shape=tsv_fasttext.shape)
            fp[:] = tsv_fasttext[:]