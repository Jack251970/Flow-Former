import sys

import argparse

import torch

from configs.submission import get_cfg as get_submission_cfg
from configs.things_eval import get_cfg as get_things_cfg
from configs.small_things_eval import get_cfg as get_small_things_cfg

from core.FlowFormer import build_flowformer

from thop import profile, clever_format

sys.path.append('core')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', help='eval benchmark')
    parser.add_argument('--small', action='store_true', help='use small model')
    args = parser.parse_args()

    cfg = None
    if args.eval == 'sintel_submission':
        cfg = get_submission_cfg()
    elif args.eval == 'kitti_submission':
        cfg = get_submission_cfg()
        cfg.latentcostformer.decoder_depth = 24
    elif args.eval == 'sintel_validation':
        if args.small:
            cfg = get_small_things_cfg()
        else:
            cfg = get_things_cfg()
    elif args.eval == 'tub_validation':
        if args.small:
            cfg = get_small_things_cfg()
        else:
            cfg = get_things_cfg()
    elif args.eval == 'kitti_validation':
        if args.small:
            cfg = get_small_things_cfg()
        else:
            cfg = get_things_cfg()
        cfg.latentcostformer.decoder_depth = 24
    else:
        print(f"EROOR: {args.eval} is not valid")
    cfg.update(vars(args))

    model = build_flowformer(cfg)

    image1 = torch.rand(1, 3, 432, 960)
    image2 = torch.rand(1, 3, 432, 960)

    macs, params = profile(model, inputs=(image1, image2))
    macs, params = clever_format([macs, params], "%.3f")

    print(macs, params)
