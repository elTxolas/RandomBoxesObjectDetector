from detectron2.data.datasets.coco import load_coco_json
import os
import cv2
import torch

def xywh_to_xminyminxmaxymax(x,y,w,h):
        xmin = x
        ymin = y
        xmax = x + w
        ymax = y + h
        return xmin, ymin, xmax, ymax

DATASET_ROOT = 'tao/'
ANN_ROOT=DATASET_ROOT
VAL_PATH = os.path.join(DATASET_ROOT, 'frames')#por qué con /train no me filtra?
VAL_JSON = os.path.join(ANN_ROOT, 'annotations-1.0/train.json')

targets_wof = load_coco_json(VAL_JSON, VAL_PATH)
targets = []
for annotation in targets_wof:
    file_name = annotation['file_name']
    if not (file_name.startswith('tao/frames/train/HACS') or file_name.startswith('tao/frames/train/AVA')):
        targets.append(annotation)
for annotation in targets: #CAMBIO DE JPEG A JPG, JPEG NO EXISTE
    file_name = annotation['file_name']
    if file_name.endswith('.jpeg'):
        annotation['file_name'] = annotation['file_name'].replace('jpeg', 'jpg')

TAO_TO_COCO_MAPPING = {91: 13, 58: 34, 621: 33, 747: 49, 118: 8, 221: 51, 95: 1, 126: 73, 1122: 79, 729: 27, 926: 48, 1117: 61, 1038: 11, 1215: 40, 276: 74, 78: 21, 1162: 75, 699: 68, 185: 55, 13: 47, 79: 59, 982: 30, 371: 60, 896: 65, 99: 14, 642: 63, 1135: 6, 717: 64, 829: 53, 1115: 70, 235: 67, 805: 0, 41: 32, 452: 10, 1155: 25, 1144: 7, 625: 43, 60: 35, 502: 23, 4: 4, 779: 12, 1001: 57, 1099: 38, 34: 24, 45: 46, 139: 45, 980: 36, 133: 39, 382: 16, 480: 29, 154: 50, 429: 20, 211: 2, 392: 54, 36: 28, 347: 41, 544: 78, 1057: 37, 1132: 9, 1097: 62, 1018: 44, 579: 17, 714: 3, 1229: 22, 229: 15, 1091: 77, 35: 26, 979: 71, 299: 66, 174: 5, 475: 42, 237: 56, 428: 72, 937: 76, 961: 18, 852: 58, 993: 31, 81: 19}
COCO_TO_OWOD_MAPPING = {0: 14, 1: 1, 2: 6, 3: 13, 4: 0, 5: 5, 6: 18, 7: 20, 8: 3, 9: 21, 10: 22, 11: 23, 12: 24, 13: 25, 14: 2, 15: 7, 16: 11, 17: 12, 18: 16, 19: 9, 20: 26, 21: 27, 22: 28, 23: 29, 24: 30, 25: 31, 26: 32, 27: 33, 28: 34, 29: 40, 30: 41, 31: 42, 32: 43, 33: 44, 34: 45, 35: 46, 36: 47, 37: 48, 38: 49, 39: 4, 40: 74, 41: 75, 42: 76, 43: 77, 44: 78, 45: 79, 46: 50, 47: 51, 48: 52, 49: 53, 50: 54, 51: 55, 52: 56, 53: 57, 54: 58, 55: 59, 56: 8, 57: 17, 58: 15, 59: 60, 60: 10, 61: 61, 62: 19, 63: 62, 64: 63, 65: 64, 66: 65, 67: 66, 68: 35, 69: 36, 70: 37, 71: 38, 72: 39, 73: 67, 74: 68, 75: 69, 76: 70, 77: 71, 78: 72, 79: 73}

for key in COCO_TO_OWOD_MAPPING:
    if COCO_TO_OWOD_MAPPING[key]>19:
        COCO_TO_OWOD_MAPPING[key]=80

for img in targets:
        for ann in img['annotations']:
            category_id = ann['category_id']
            if category_id in TAO_TO_COCO_MAPPING:
                ann['category_id'] = TAO_TO_COCO_MAPPING[category_id]
    
for img in targets:
    for ann in img['annotations']:
        category_id = ann['category_id']
        if category_id in COCO_TO_OWOD_MAPPING:
            ann['category_id'] = COCO_TO_OWOD_MAPPING[category_id] #ANOTACIONES EN FORMATO PAPER
        else:
            ann['category_id'] = 80 #algunos no están en el mapping
for img in targets:
    for ann in img['annotations']:
        x,y,w,h = ann['bbox']
        ann['bbox'] = xywh_to_xminyminxmaxymax(x,y,w,h)#cambio las bboxes

abs_dir = 'tao/frames/train'
total_videos = []
for directory in sorted(os.listdir(abs_dir)):
    for dir in sorted(os.listdir(abs_dir + '/' + directory)):
        image_root = abs_dir + '/' + directory + '/' + dir
        image_files = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.jpeg')])
        total_videos.append(image_files)
id_diff = 0
i = 0
for image_files in total_videos:
    for id, file in enumerate(image_files):
        found = False
        image = cv2.imread(file)
        for target in targets:
            if target['file_name'] == file:
                found = True
                target['image_id'] = id + id_diff
                break
        if not found:
            targets.append({
                'file_name': file,
                'height': image.shape[0],
                'width': image.shape[1],
                'image_id': id + id_diff,
                'annotations': None  
            })

    id_diff += id + 1
    i += 1
    print(str(i) + '/' + str(len(total_videos)))
targets.sort(key=lambda x: x['image_id'])
torch.save(targets, 'targets_with_images_ALL.pkl')