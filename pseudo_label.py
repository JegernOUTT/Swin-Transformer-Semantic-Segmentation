import uuid
from argparse import ArgumentParser
from logging import warning
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import torch
from dssl_dl_datasets import Hasty
from dssl_dl_utils import ImageAnnotations, Size2D, BitmapMask, MaskType, Instance, Bbox
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


def _postprocess_seg_map(seg_map: np.ndarray) -> Dict[str, BitmapMask]:
    def _filter_mask_by_indexes(indexes):
        result = np.zeros_like(seg_map, dtype=np.uint8)
        for idx in indexes:
            result[seg_map == idx] = 255
        return BitmapMask(result, image_size=Size2D(width=seg_map.shape[1], height=seg_map.shape[0]),
                          mask_type=MaskType.SemanticSegmentation)

    road_indexes = [6, 11, 52, 61]
    building_indexes = [0, 1, 25, 48, 79, 84, 86]
    fence_indexes = [32, 38]
    vegetation_indexes = [4, 17, 66, 72]
    terrain_indexes = [3, 13, 28, 29, 46, 94, 101, 140]
    sky_indexes = [2, 5]
    stuff_indexes = [7, 8, 10, 15, 18, 19, 22, 23, 24, 27, 30, 31, 33, 34, 35, 36, 37,
                     39, 40, 41, 42, 43, 44, 45, 47, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59,
                     62, 63, 64, 65, 67, 69, 70, 71, 73, 74, 75, 76, 77, 78, 81, 82, 85, 87,
                     88, 89, 90, 92, 93, 95, 96, 97, 98, 99, 100, 103, 104, 105, 106, 107, 108,
                     110, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                     127, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 141, 142, 143,
                     144, 145, 146, 147, 148, 149]
    water_indexes = [21, 26, 60, 109, 113, 128]
    grass_indexes = [9]
    mountain_indexes = [16, 68]
    ignore_indexes = [12, 20, 80, 83, 91, 102, 126]

    return {
        'road': _filter_mask_by_indexes(road_indexes),
        'building': _filter_mask_by_indexes(building_indexes),
        'fence': _filter_mask_by_indexes(fence_indexes),
        'vegetation': _filter_mask_by_indexes(vegetation_indexes),
        'terrain': _filter_mask_by_indexes(terrain_indexes),
        'mountain': _filter_mask_by_indexes(mountain_indexes),
        'sky': _filter_mask_by_indexes(sky_indexes),
        'stuff': _filter_mask_by_indexes(stuff_indexes),
        'water': _filter_mask_by_indexes(water_indexes),
        'grass': _filter_mask_by_indexes(grass_indexes),
    }


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


def _infer(args, data: List[ImageAnnotations], wait_key=1) -> Dict[str, np.ndarray]:
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    model = convert_batchnorm(model)
    model.eval()

    # data = [info for info in data if info.image_info.meta['image_status'].lower() == 'new']
    if len(data) == 0:
        warning(f'No files found in {args.path}')
        return {}
    seg_maps = {}
    cv2.namedWindow('demo', cv2.WINDOW_KEEPRATIO)
    for info in tqdm(data, desc='Images inference'):
        filename = str(info.image_info.filename)
        frame = cv2.imread(filename)
        with torch.no_grad():
            result = inference_segmentor(model, frame)
        seg_maps[filename] = _postprocess_seg_map(result[0])
        img = model.show_result(frame, result, palette=get_palette(args.palette), show=False)
        cv2.imshow('demo', img)
        ch = cv2.waitKey(wait_key) & 0xFF
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

    return seg_maps


def _fill_ds(data, seg_maps):
    # data = [info for info in data if info.image_info.meta['image_status'].lower() == 'new']

    for info in data:
        filename = str(info.image_info.filename)
        segs = seg_maps[filename]
        info.instances.clear()
        for cat_name, seg in segs.items():
            info.instances.append(Instance(
                bboxes=[Bbox.from_xyxy((0., 0., 1., 1.))], masks=[seg],
                meta=dict(attributes={}, z_index=1, category_name=cat_name,
                          id=str(uuid.uuid4()))))


def main():
    parser = ArgumentParser()
    parser.add_argument('path', type=Path, help='Path with images')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cpu', help='Device used for inference')
    parser.add_argument('--palette',
                        default='ade',
                        help='Color palette used for segmentation map')
    args = parser.parse_args()
    torch.set_num_threads(16)

    hasty_ds = Hasty()
    data = hasty_ds.load(args.path / 'base.json', args.path)
    seg_maps = _infer(args, data)
    _fill_ds(data, seg_maps)

    parts = len(data)
    for i in range(len(data) // parts):
        hasty_ds.dump(data[i * parts: (i + 1) * parts], args.path / 'new_anns',
                      annotations_filename=f'annotations_{i}', clone_images=False, group_by=False)


if __name__ == '__main__':
    main()
