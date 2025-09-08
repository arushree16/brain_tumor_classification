import argparse
import os
import subprocess
import sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cmd', required=True, choices=['train','eval','gradcam'])
    p.add_argument('--data_root', default='data_root')
    p.add_argument('--weights', default=None)
    args, unknown = p.parse_known_args()

    python_exe = sys.executable  # ensures same interpreter

    if args.cmd == 'train':
        cmd = [python_exe, 'train_sam.py', '--data_root', args.data_root] + unknown
    elif args.cmd == 'eval':
        if args.weights is None:
            raise ValueError('weights required for eval')
        cmd = [python_exe, 'evaluate.py', '--data_root', args.data_root,
               '--weights', args.weights] + unknown
    elif args.cmd == 'gradcam':
        if args.weights is None:
            raise ValueError('weights required for gradcam')
        cmd = [python_exe, 'explainability/grad_cam.py',
               '--weights', args.weights, '--data_root', args.data_root] + unknown

    print('Running:', ' '.join(cmd))
    ret = subprocess.call(cmd)
    if ret != 0:
        raise RuntimeError(f"Command failed with exit code {ret}")

if __name__ == '__main__':
    main()
