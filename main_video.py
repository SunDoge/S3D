import argparse
import logging
import os
from dataclasses import dataclass
from typing import List

import decord
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms._transforms_video import (NormalizeVideo,
                                                      ToTensorVideo)
from typed_args import TypedArgs, add_argument

from model import S3D

_logger = logging.getLogger(__name__)
decord.bridge.set_bridge('torch')


@dataclass
class Args(TypedArgs):
    video_path: str = add_argument(help='input video path')
    weight: str = add_argument('-w', '--weight', help='weight file')
    cpu: bool = add_argument('--cpu', action='store_true')
    frame_range: List[int] = add_argument(
        '-f', '--frame-range', nargs=argparse.ZERO_OR_MORE)


def load_weight(model, file_weight: str):
    # load the weight file and copy the parameters
    if os.path.isfile(file_weight):
        print('loading weight file')
        weight_dict = torch.load(file_weight, map_location='cpu')
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    print(' size? ' + name, param.size(),
                          model_dict[name].size())
            else:
                print(' name? ' + name)

        print(' loaded')
    else:
        print('weight file?')


def get_transform():
    trans = T.Compose([
        ToTensorVideo(),
        NormalizeVideo(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            inplace=True,
        )
    ])
    return trans


def main():
    logging.basicConfig(level=logging.INFO)
    args = Args.from_args()
    _logger.info(args)

    torch.set_grad_enabled(False)

    device = torch.device(
        'cuda' if torch.cuda.is_available()
        and not args.cpu else 'cpu'
    )

    class_names = [c.strip() for c in open('./label_map.txt')]
    num_class = 400

    model = S3D(num_class)

    load_weight(model, args.weight)

    model.to(device)
    model.eval()

    vr = decord.VideoReader(args.video_path)
    frame_range = [
        0, len(vr)] if args.frame_range is None else args.frame_range
    _logger.info('frame range: %s', frame_range)
    batch = vr.get_batch(np.arange(*frame_range))
    transform = get_transform()
    video = transform(batch)

    video = video.to(device).unsqueeze(0)

    _logger.info('video: %s', video.shape)

    logits = model(video)[0]

    preds = torch.softmax(logits, 0).cpu().numpy()

    _logger.info('preds: %s', preds.shape)
    sorted_indices = np.argsort(preds)[::-1][:5]

    print('\nTop 5 classes ... with probability')

    for idx in sorted_indices:
        print(class_names[idx], '...', preds[idx])


if __name__ == "__main__":
    main()
