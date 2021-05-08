from argparse import ArgumentParser
from itertools import chain
from logging import warning
from pathlib import Path

import cv2
from dssl_dl_utils.utils.capture import VideoCaptureWrapper

from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette


def main():
    parser = ArgumentParser()
    parser.add_argument('path', type=Path, help='Path to videofiles')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--palette',
                        default='ade',
                        help='Color palette used for segmentation map')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    filenames = chain.from_iterable([args.path.rglob(f'*.{ext}') for ext in {'avi', 'mp4'}])
    filenames = list(filenames)
    if len(filenames) == 0:
        warning(f'No files found in {args.path}')
        return

    cv2.namedWindow('demo', cv2.WINDOW_KEEPRATIO)
    for filename in filenames:
        capture = VideoCaptureWrapper(filename)
        for frame, _, _ in capture:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = inference_segmentor(model, frame)
            img = model.show_result(frame, result, palette=get_palette(args.palette), show=False)
            cv2.imshow('demo', img)
            ch = cv2.waitKey(1) & 0xFF
            if ch == 27 or ch == ord('q') or ch == ord('Q'):
                break


if __name__ == '__main__':
    main()
