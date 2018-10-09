import torch
import argparse

from cinic10.model import wrn_22
from cinic10.model import get_learner

cuda = True if torch.cuda.is_available() else False


def main(batch_size=128):
    learn = get_learner(arch=wrn_22(), batch_size=batch_size)
    learn.clip = 1e-1
    learn.fit(lrs=1.5, n_cycle=1, wds=1e-4, cycle_len=20, use_clr_beta=(12, 15, 0.95, 0.85))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    batch_size = args.batch_size

    main(batch_size=batch_size)
