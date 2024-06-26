import torch
from detectron2.data.datasets.coco import load_coco_json
import cv2
from detectron2.utils.visualizer import Visualizer

all_predictions_500 = torch.load('all_predictions_500.pth')
all_predictions_20 = torch.load('all_predictions_20?.pth')

all_targets_wof = load_coco_json('tao/annotations-1.0/train.json', 'tao/frames')

all_targets = []

for annotation in all_targets_wof:
    file_name = annotation['file_name']
    if not (file_name.startswith('tao/frames/train/HACS') or file_name.startswith('tao/frames/train/AVA')):
        all_targets.append(annotation)

for annotation in all_targets: #CAMBIO DE JPEG A JPG, JPEG NO EXISTE
    file_name = annotation['file_name']
    if file_name.endswith('.jpeg'):
        annotation['file_name'] = annotation['file_name'].replace('jpeg', 'jpg')

TAO_TO_COCO_MAPPING = {91: 13, 58: 34, 621: 33, 747: 49, 118: 8, 221: 51, 95: 1, 126: 73, 1122: 79, 729: 27, 926: 48, 1117: 61, 1038: 11, 1215: 40, 276: 74, 78: 21, 1162: 75, 699: 68, 185: 55, 13: 47, 79: 59, 982: 30, 371: 60, 896: 65, 99: 14, 642: 63, 1135: 6, 717: 64, 829: 53, 1115: 70, 235: 67, 805: 0, 41: 32, 452: 10, 1155: 25, 1144: 7, 625: 43, 60: 35, 502: 23, 4: 4, 779: 12, 1001: 57, 1099: 38, 34: 24, 45: 46, 139: 45, 980: 36, 133: 39, 382: 16, 480: 29, 154: 50, 429: 20, 211: 2, 392: 54, 36: 28, 347: 41, 544: 78, 1057: 37, 1132: 9, 1097: 62, 1018: 44, 579: 17, 714: 3, 1229: 22, 229: 15, 1091: 77, 35: 26, 979: 71, 299: 66, 174: 5, 475: 42, 237: 56, 428: 72, 937: 76, 961: 18, 852: 58, 993: 31, 81: 19}
COCO_TO_OWOD_MAPPING = {0: 14, 1: 1, 2: 6, 3: 13, 4: 0, 5: 5, 6: 18, 7: 20, 8: 3, 9: 21, 10: 22, 11: 23, 12: 24, 13: 25, 14: 2, 15: 7, 16: 11, 17: 12, 18: 16, 19: 9, 20: 26, 21: 27, 22: 28, 23: 29, 24: 30, 25: 31, 26: 32, 27: 33, 28: 34, 29: 40, 30: 41, 31: 42, 32: 43, 33: 44, 34: 45, 35: 46, 36: 47, 37: 48, 38: 49, 39: 4, 40: 74, 41: 75, 42: 76, 43: 77, 44: 78, 45: 79, 46: 50, 47: 51, 48: 52, 49: 53, 50: 54, 51: 55, 52: 56, 53: 57, 54: 58, 55: 59, 56: 8, 57: 17, 58: 15, 59: 60, 60: 10, 61: 61, 62: 19, 63: 62, 64: 63, 65: 64, 66: 65, 67: 66, 68: 35, 69: 36, 70: 37, 71: 38, 72: 39, 73: 67, 74: 68, 75: 69, 76: 70, 77: 71, 78: 72, 79: 73}

for key in COCO_TO_OWOD_MAPPING:
    if COCO_TO_OWOD_MAPPING[key]>19:
        COCO_TO_OWOD_MAPPING[key]=80

for img in all_targets:
        for ann in img['annotations']:
            category_id = ann['category_id']
            if category_id in TAO_TO_COCO_MAPPING:
                ann['category_id'] = TAO_TO_COCO_MAPPING[category_id]
    
for img in all_targets:
    for ann in img['annotations']:
        category_id = ann['category_id']
        if category_id in COCO_TO_OWOD_MAPPING:
            ann['category_id'] = COCO_TO_OWOD_MAPPING[category_id] #ANOTACIONES EN FORMATO PAPER
        else:
            ann['category_id'] = 80 #algunos no están en el mapping, así bien? 

# #DRAW ANOTATIONS
            
# i = 0
# for d in all_targets:
#     img = cv2.imread(d["file_name"])
#     parts = d['file_name'].split('/')
#     if parts[3] == 'BDD':
#         i += 1
#         visualizer = Visualizer(img[:, :, ::-1], scale=1.5)
#         vis = visualizer.draw_dataset_dict(d)
#         cv2.imwrite('outTAO/TAOanotations/BDD/' + str(i) + '.jpg', vis.get_image()[:, :, ::-1])

# i = 0
# for d in all_targets:
#     img = cv2.imread(d["file_name"])
#     parts = d['file_name'].split('/')
#     if parts[3] == 'Charades':
#         i += 1
#         visualizer = Visualizer(img[:, :, ::-1], scale=1.5)
#         vis = visualizer.draw_dataset_dict(d)
#         cv2.imwrite('outTAO/TAOanotations/Charades/' + str(i) + '.jpg', vis.get_image()[:, :, ::-1])

# i = 0
# for d in all_targets:
#     img = cv2.imread(d["file_name"])
#     parts = d['file_name'].split('/')
#     if parts[3] == 'LaSOT':
#         i += 1
#         visualizer = Visualizer(img[:, :, ::-1], scale=1.5)
#         vis = visualizer.draw_dataset_dict(d)
#         cv2.imwrite('outTAO/TAOanotations/LaSOT/' + str(i) + '.jpg', vis.get_image()[:, :, ::-1])

# i = 0
# for d in all_targets:
#     img = cv2.imread(d["file_name"])
#     parts = d['file_name'].split('/')
#     if parts[3] == 'YFCC100M':
#         i += 1
#         visualizer = Visualizer(img[:, :, ::-1], scale=1.5)
#         vis = visualizer.draw_dataset_dict(d)
#         cv2.imwrite('outTAO/TAOanotations/YFCC100M/' + str(i) + '.jpg', vis.get_image()[:, :, ::-1])


#FILTER OUT PREDICTIONS WITH A 0.2 SCORE THRESHOLD
from detectron2.structures import Instances, Boxes

# Filter based on the score threshold
new_out_pred_500 = []
threshold = 0.2
for i,pred in enumerate(all_predictions_500):
    above_threshold_mask = pred['instances'].scores > threshold
    num_instances = above_threshold_mask.sum().item()

    pred_boxes_at = pred['instances'].pred_boxes[above_threshold_mask]
    scores_at = pred['instances'].scores[above_threshold_mask]
    pred_classes_at = pred['instances'].pred_classes[above_threshold_mask]

    image_size = (all_targets[i]['height'], all_targets[i]['width'])
    new_out_pred_500.append(Instances(
        image_size=image_size, pred_boxes=pred_boxes_at, scores=scores_at, pred_classes=pred_classes_at
        ))
#DRAW 500 BBOXES (PROPOSALS) PREDICTIONS
        
j = 0
for i,pred in enumerate(new_out_pred_500):
    im = cv2.imread(all_targets[i]['file_name'])
    parts = all_targets[i]['file_name'].split('/')
    if parts[3] == 'ArgoVerse':
        j += 1
        v = Visualizer(im[:,:,::-1], scale=1.5)
        v = v.draw_instance_predictions(pred.to('cpu'))
        img = v.get_image()[:, :, ::-1]
        cv2.imwrite('outTAO/TAOpredictions500/ArgoVerse/' + str(j) + '.jpg', img)

j = 0
for i,pred in enumerate(new_out_pred_500):
    im = cv2.imread(all_targets[i]['file_name'])
    parts = all_targets[i]['file_name'].split('/')
    if parts[3] == 'BDD':
        j += 1
        v = Visualizer(im[:,:,::-1], scale=1.5)
        v = v.draw_instance_predictions(pred.to('cpu'))
        img = v.get_image()[:, :, ::-1]
        cv2.imwrite('outTAO/TAOpredictions500/BDD/' + str(j) + '.jpg', img)

j = 0
for i,pred in enumerate(new_out_pred_500):
    im = cv2.imread(all_targets[i]['file_name'])
    parts = all_targets[i]['file_name'].split('/')
    if parts[3] == 'LaSOT':
        j += 1
        v = Visualizer(im[:,:,::-1], scale=1.5)
        v = v.draw_instance_predictions(pred.to('cpu'))
        img = v.get_image()[:, :, ::-1]
        cv2.imwrite('outTAO/TAOpredictions500/LaSOT/' + str(j) + '.jpg', img)

new_out_pred_20 = []
threshold = 0.2
for i,pred in enumerate(all_predictions_20):
    above_threshold_mask = pred['instances'].scores > threshold
    num_instances = above_threshold_mask.sum().item()

    pred_boxes_at = pred['instances'].pred_boxes[above_threshold_mask]
    scores_at = pred['instances'].scores[above_threshold_mask]
    pred_classes_at = pred['instances'].pred_classes[above_threshold_mask]

    image_size = (all_targets[i]['height'], all_targets[i]['width'])
    new_out_pred_20.append(Instances(
        image_size=image_size, pred_boxes=pred_boxes_at, scores=scores_at, pred_classes=pred_classes_at
        ))
#DRAW 500 BBOXES (PROPOSALS) PREDICTIONS
        
j = 0
for i,pred in enumerate(new_out_pred_20):
    im = cv2.imread(all_targets[i]['file_name'])
    parts = all_targets[i]['file_name'].split('/')
    if parts[3] == 'ArgoVerse':
        j += 1
        v = Visualizer(im[:,:,::-1], scale=1.5)
        v = v.draw_instance_predictions(pred.to('cpu'))
        img = v.get_image()[:, :, ::-1]
        cv2.imwrite('outTAO/TAOpredictions500/ArgoVerse/' + str(j) + '.jpg', img)

j = 0
for i,pred in enumerate(new_out_pred_20):
    im = cv2.imread(all_targets[i]['file_name'])
    parts = all_targets[i]['file_name'].split('/')
    if parts[3] == 'BDD':
        j += 1
        v = Visualizer(im[:,:,::-1], scale=1.5)
        v = v.draw_instance_predictions(pred.to('cpu'))
        img = v.get_image()[:, :, ::-1]
        cv2.imwrite('outTAO/TAOpredictions500/BDD/' + str(j) + '.jpg', img)

j = 0
for i,pred in enumerate(new_out_pred_20):
    im = cv2.imread(all_targets[i]['file_name'])
    parts = all_targets[i]['file_name'].split('/')
    if parts[3] == 'LaSOT':
        j += 1
        v = Visualizer(im[:,:,::-1], scale=1.5)
        v = v.draw_instance_predictions(pred.to('cpu'))
        img = v.get_image()[:, :, ::-1]
        cv2.imwrite('outTAO/TAOpredictions500/LaSOT/' + str(j) + '.jpg', img)