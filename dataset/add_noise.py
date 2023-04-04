import os
from argparse import ArgumentParser
from typing import Tuple

import numpy as np
import scipy
from sklearn.datasets import dump_svmlight_file, load_svmlight_file


def parse_args():
    parser = ArgumentParser('add_noise')
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--level", type=float, required=True, help='{1e-1, 1e-2, 1e-3, 1e-4}')
    parser.add_argument("--type", type=str, required=True, help='{feature, label}')
    parser.add_argument("--svm-file-path", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--save-name", type=str, required=True)
    return parser.parse_args()


def _svm_hint_type(ds) -> Tuple[scipy.sparse.csr_matrix, np.ndarray, np.ndarray]:
    return ds


def add_feature_noise(svm_file_path: str, sigma: float, save_dir: str, save_name: str):
    x, y, query_ids = _svm_hint_type(load_svmlight_file(svm_file_path, query_id=True))
    noise = sigma * np.random.randn(*x.shape)
    x += noise
    os.makedirs(save_dir, exist_ok=True)
    dump_svmlight_file(np.asarray(x), y, os.path.join(save_dir, save_name), query_id=query_ids)


def add_label_noise(svm_file_path: str, sigma: float, save_dir: str, save_name: str):
    raise NotImplementedError


def main():
    args = parse_args()
    assert isinstance(args.seed, int)
    assert isinstance(args.level, float)
    if args.type == 'feature':
        noise_fn = add_feature_noise
    elif args.type == 'label':
        noise_fn = add_label_noise
    else:
        raise ValueError(f'Invalid noise type {args.type}')
    print(f'Add {args.level} {args.type} noise ({args.seed}) to {args.svm_file_path}, save to {args.save_dir}/{args.save_name}')

    np.random.seed(args.seed)
    noise_fn(args.svm_file_path, args.level, args.save_dir, args.save_name)


if __name__ == '__main__':
    main()
