import uuid
from argparse import ArgumentParser
from itertools import chain
from logging import warning
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import torch
from dssl_dl_datasets import Hasty, Internal
from dssl_dl_utils import ImageAnnotations, Size2D, BitmapMask, MaskType, Instance, Bbox, ImageInfo
from tqdm import tqdm

from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette

CLASSNAMES = (
    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
    'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
    'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
    'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
    'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
    'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
    'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
    'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
    'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
    'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
    'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
    'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
    'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
    'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
    'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
    'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
    'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
    'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
    'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
    'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
    'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
    'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
    'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
    'clock', 'flag'
)


def _postprocess_seg_map(seg_map: np.ndarray) -> List[BitmapMask]:
    def _get_mask_by_index(idx, category_name):
        result = np.zeros_like(seg_map, dtype=np.uint8)
        result[seg_map == idx] = 255
        return BitmapMask(result, image_size=Size2D(width=seg_map.shape[1], height=seg_map.shape[0]),
                          category_name=category_name, mask_type=MaskType.SemanticSegmentation)

    return [_get_mask_by_index(idx, cls_name)
            for idx, cls_name in enumerate(CLASSNAMES)]


def convert_batchnorm(module, process_group=None):
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features,
                                             module.eps, module.momentum,
                                             module.affine,
                                             module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, convert_batchnorm(child, process_group))
    del module
    return module_output


def _infer(args, filenames: List[ImageAnnotations], wait_key=1) -> Dict[str, np.ndarray]:
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    model = convert_batchnorm(model)
    model.eval()

    if len(filenames) == 0:
        warning(f'No files found in {args.path}')
        return {}

    # cv2.namedWindow('demo', cv2.WINDOW_KEEPRATIO)
    for filename in tqdm(filenames, desc='Images inference'):
        frame = cv2.imread(str(filename))
        with torch.no_grad():
            result = inference_segmentor(model, frame)
        yield ImageInfo(filename=filename, size=Size2D(width=frame.shape[1], height=frame.shape[0])),\
              _postprocess_seg_map(result[0])
        # img = model.show_result(frame, result, palette=get_palette(args.palette), show=False)
        # cv2.imshow('demo', img)
        # ch = cv2.waitKey(wait_key) & 0xFF
        # if ch == 27 or ch == ord('q') or ch == ord('Q'):
        #     break


def _fill_ds(info, bitmaps):
    return [
        ImageAnnotations(info, [Instance(masks=[bitmap]) for bitmap in bitmaps])
    ]


def main():
    parser = ArgumentParser()
    parser.add_argument('path', type=Path, help='Path with images')
    parser.add_argument('ann_path', type=Path, help='Path with annotations')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda', help='Device used for inference')
    parser.add_argument('--palette',
                        default='ade',
                        help='Color palette used for segmentation map')
    args = parser.parse_args()
    torch.set_num_threads(16)

    filenames = chain.from_iterable([args.path.rglob(f'*.{ext}')
                                     for ext in {'jpg', 'jpeg', 'png'}])
    filenames = list(filenames)

    for info, bitmaps in _infer(args, filenames):
        img_anns = _fill_ds(info, bitmaps)
        Internal.dump(img_anns, args.ann_path, clone_images=False,
                      annotations_filename=info.filename.stem, group_by=False)


if __name__ == '__main__':
    main()
