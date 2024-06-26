"""
RandBox Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import os
import itertools
import weakref
from typing import Any, Dict, List, Set
import logging
from collections import OrderedDict

import torch
from fvcore.nn.precise_bn import get_bn_modules

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.engine import default_argument_parser, default_setup, launch, create_ddp_model, \
    AMPTrainer, SimpleTrainer, hooks
from detectron2.evaluation import COCOEvaluator, LVISEvaluator, verify_results
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.modeling import build_model
from detectron2.engine.my_defaults import DefaultTrainer#mi default
from randbox import RandBoxDatasetMapper, add_RandBox_config, RandBoxWithTTA
from randbox.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer


from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from randbox.my_pascal_voc_evaluation import PascalVOCDetectionEvaluator

class Register:
    """用于注册自己的数据集"""
    CLASS_NAMES = ['__background__', '0']  # 保留 background 类

    def __init__(self,dataset_root):
        self.CLASS_NAMES = Register.CLASS_NAMES or ['__background__', ]
        # 数据集路径
        self.DATASET_ROOT = dataset_root
        # ANN_ROOT = os.path.join(self.DATASET_ROOT, 'COCOformat')
        self.ANN_ROOT = self.DATASET_ROOT

        # self.TRAIN_PATH = os.path.join(self.DATASET_ROOT, 'images/train')
        self.VAL_PATH = os.path.join(self.DATASET_ROOT, 'frames')#por qué con /train no me filtra?
        # self.TRAIN_JSON = os.path.join(self.ANN_ROOT, 'annotations/train.json')
        self.VAL_JSON = os.path.join(self.ANN_ROOT, 'annotations-1.0/train.json')
        # VAL_JSON = os.path.join(self.ANN_ROOT, 'test.json')

        # 声明数据集的子集
        self.PREDEFINED_SPLITS_DATASET = {
            #"my_train": (self.TRAIN_PATH, self.TRAIN_JSON),
            "my_val": (self.VAL_PATH, self.VAL_JSON),
        }

    def register_dataset(self):
        """
        purpose: register all splits of datasets with PREDEFINED_SPLITS_DATASET
        注册数据集（这一步就是将自定义数据集注册进Detectron2）
        """
        for key, (image_root, json_file) in self.PREDEFINED_SPLITS_DATASET.items():
            targets = self.register_dataset_instances(self,name=key,
                                            json_file=json_file,
                                            image_root=image_root)#añado targets, para llevarlos al main y meterlo en evaluator
        return targets
    @staticmethod
    def register_dataset_instances(self, name, json_file, image_root):
        """
        purpose: register datasets to DatasetCatalog,
                 register metadata to MetadataCatalog and set attribute
        注册数据集实例，加载数据集中的对象实例
        """
        #Añado para filtrar y corregir bboxes de anotaciones
        # targets_wof = load_coco_json(json_file, image_root, name)#porque las id con -1?
        if self.DATASET_ROOT == 'tao/':
            # targets_wof = load_coco_json(json_file, image_root)#si no pongo el name, no me hace el -1, CAMBIO
            # targets = []
            # # for annotation in targets_wof: HAGO SOLO CON UN FOLDER
            # #     file_name = annotation['file_name']
            # #     if not (file_name.startswith('tao/frames/train/HACS') or file_name.startswith('tao/frames/train/AVA')):
            # #         targets.append(annotation)
            # for annotation in targets_wof: 
            #     file_name = annotation['file_name']
            #     if file_name.startswith('tao/frames/train/LaSOT/airplane-4'):
            #         targets.append(annotation)
            # for annotation in targets: #CAMBIO DE JPEG A JPG, JPEG NO EXISTE
            #     file_name = annotation['file_name']
            #     if file_name.endswith('.jpeg'):
            #         annotation['file_name'] = annotation['file_name'].replace('jpeg', 'jpg')

            # TAO_TO_COCO_MAPPING = {91: 13, 58: 34, 621: 33, 747: 49, 118: 8, 221: 51, 95: 1, 126: 73, 1122: 79, 729: 27, 926: 48, 1117: 61, 1038: 11, 1215: 40, 276: 74, 78: 21, 1162: 75, 699: 68, 185: 55, 13: 47, 79: 59, 982: 30, 371: 60, 896: 65, 99: 14, 642: 63, 1135: 6, 717: 64, 829: 53, 1115: 70, 235: 67, 805: 0, 41: 32, 452: 10, 1155: 25, 1144: 7, 625: 43, 60: 35, 502: 23, 4: 4, 779: 12, 1001: 57, 1099: 38, 34: 24, 45: 46, 139: 45, 980: 36, 133: 39, 382: 16, 480: 29, 154: 50, 429: 20, 211: 2, 392: 54, 36: 28, 347: 41, 544: 78, 1057: 37, 1132: 9, 1097: 62, 1018: 44, 579: 17, 714: 3, 1229: 22, 229: 15, 1091: 77, 35: 26, 979: 71, 299: 66, 174: 5, 475: 42, 237: 56, 428: 72, 937: 76, 961: 18, 852: 58, 993: 31, 81: 19}
            # COCO_TO_OWOD_MAPPING = {0: 14, 1: 1, 2: 6, 3: 13, 4: 0, 5: 5, 6: 18, 7: 20, 8: 3, 9: 21, 10: 22, 11: 23, 12: 24, 13: 25, 14: 2, 15: 7, 16: 11, 17: 12, 18: 16, 19: 9, 20: 26, 21: 27, 22: 28, 23: 29, 24: 30, 25: 31, 26: 32, 27: 33, 28: 34, 29: 40, 30: 41, 31: 42, 32: 43, 33: 44, 34: 45, 35: 46, 36: 47, 37: 48, 38: 49, 39: 4, 40: 74, 41: 75, 42: 76, 43: 77, 44: 78, 45: 79, 46: 50, 47: 51, 48: 52, 49: 53, 50: 54, 51: 55, 52: 56, 53: 57, 54: 58, 55: 59, 56: 8, 57: 17, 58: 15, 59: 60, 60: 10, 61: 61, 62: 19, 63: 62, 64: 63, 65: 64, 66: 65, 67: 66, 68: 35, 69: 36, 70: 37, 71: 38, 72: 39, 73: 67, 74: 68, 75: 69, 76: 70, 77: 71, 78: 72, 79: 73}

            # for key in COCO_TO_OWOD_MAPPING:
            #     if COCO_TO_OWOD_MAPPING[key]>19:
            #         COCO_TO_OWOD_MAPPING[key]=80

            # for img in targets:
            #         for ann in img['annotations']:
            #             category_id = ann['category_id']
            #             if category_id in TAO_TO_COCO_MAPPING:
            #                 ann['category_id'] = TAO_TO_COCO_MAPPING[category_id]
                
            # for img in targets:
            #     for ann in img['annotations']:
            #         category_id = ann['category_id']
            #         if category_id in COCO_TO_OWOD_MAPPING:
            #             ann['category_id'] = COCO_TO_OWOD_MAPPING[category_id] #ANOTACIONES EN FORMATO PAPER
            #         else:
            #             ann['category_id'] = 80 #algunos no están en el mapping
            # for img in targets:
            #     for ann in img['annotations']:
            #         x,y,w,h = ann['bbox']
            #         ann['bbox'] = Register.xywh_to_xminyminxmaxymax(x,y,w,h)#cambio las bboxes
            # # hasta aquí
            targets_wof = torch.load('targets_with_images_ALL.pkl')
            targets = []
            for tar in targets_wof:
                if tar['file_name'].startswith('tao/frames/train/LaSOT'):
                    targets.append(tar)
            # targets = targets_wof
            DatasetCatalog.register(name, lambda: targets)
            MetadataCatalog.get(name).set(json_file=json_file,
                                        image_root=image_root,
                                        evaluator_type="coco")
        return targets#añado
    #Añado para cambiar formato bboxes
    def xywh_to_xminyminxmaxymax(x,y,w,h):
        xmin = x
        ymin = y
        xmax = x + w
        ymax = y + h
        return xmin, ymin, xmax, ymax
    #hasta aquí
    def plain_register_dataset(self):
        """注册数据集和元数据"""
        # 训练集
        DatasetCatalog.register("my_train", lambda: load_coco_json(self.TRAIN_JSON, self.TRAIN_PATH))
        MetadataCatalog.get("my_train").set(thing_classes=self.CLASS_NAMES,  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
                                                 evaluator_type='coco',  # 指定评估方式
                                                 json_file=self.TRAIN_JSON,
                                                 image_root=self.TRAIN_PATH)
        # DatasetCatalog.register("coco_my_val", lambda: load_coco_json(VAL_JSON, VAL_PATH, "coco_2017_val"))
        # 验证/测试集
        DatasetCatalog.register("my_val", lambda: self.VAL_JSON, self.VAL_PATH)
        MetadataCatalog.get("my_val").set(thing_classes=self.CLASS_NAMES,  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
                                               evaluator_type='coco',  # 指定评估方式
                                               json_file=self.VAL_JSON,
                                               image_root=self.VAL_PATH)

    def checkout_dataset_annotation(self, name="my_val"):
        """
        查看数据集标注，可视化检查数据集标注是否正确，
        这个也可以自己写脚本判断，其实就是判断标注框是否超越图像边界
        可选择使用此方法
        """
        # dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH, name)
        dataset_dicts = load_coco_json(self.TRAIN_JSON, self.TRAIN_PATH)
        print(len(dataset_dicts))
        for i, d in enumerate(dataset_dicts, 0):
            # print(d)
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(name), scale=1.5)
            vis = visualizer.draw_dataset_dict(d)
            # cv2.imshow('show', vis.get_image()[:, :, ::-1])
            cv2.imwrite('out/' + str(i) + '.jpg', vis.get_image()[:, :, ::-1])
            # cv2.waitKey(0)
            if i == 200:
                break




class Trainer(DefaultTrainer):
    """ Extension of the Trainer class adapted to RandBox. """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super(DefaultTrainer, self).__init__()  # call grandfather's `__init__` while avoid father's `__init()`
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        ########## EMA ############
        kwargs = {
            'trainer': weakref.proxy(self),
        }
        kwargs.update(may_get_ema_checkpointer(cfg, model))
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
            # trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        # setup EMA
        may_build_model_ema(cfg, model)
        return model

    @classmethod
    def build_evaluator(cls, targets, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if 'lvis' in dataset_name:
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        else:
#             return COCOEvaluator(dataset_name, cfg, True, output_folder)
            return PascalVOCDetectionEvaluator(targets=targets, dataset_name=dataset_name, cfg=cfg)#añado targets
            

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = RandBoxDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def ema_test(cls, cfg, model, targets, evaluators=None):#añado targets, para llevarlos al evaluator
        # model with ema weights
        logger = logging.getLogger("detectron2.trainer")
        if cfg.MODEL_EMA.ENABLED:
            logger.info("Run evaluation with EMA.")
            with apply_model_ema_and_restore(model):
                results = cls.test(cfg, model, evaluators=evaluators)
        else:
            results = cls.test(cfg, model, targets, evaluators=evaluators)#mi defaults
        return results

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = RandBoxWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        if cfg.MODEL_EMA.ENABLED:
            cls.ema_test(cfg, model, evaluators)
        else:
            res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            EMAHook(self.cfg, self.model) if cfg.MODEL_EMA.ENABLED else None,  # EMA hook
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_RandBox_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    data_Register=Register('tao/')
    #data_Register=Register('./datasets/'+args.task)
    targets = data_Register.register_dataset()
    if args.eval_only:
        model = Trainer.build_model(cfg)
        kwargs = may_get_ema_checkpointer(cfg, model)
        if cfg.MODEL_EMA.ENABLED:
            EMADetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(cfg.MODEL.WEIGHTS,
                                                                                              resume=args.resume)
        else:
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(cfg.MODEL.WEIGHTS,
                                                                                           resume=args.resume)
        res = Trainer.ema_test(cfg, model, targets=targets)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser0 = default_argument_parser()
    parser0.add_argument("--task", default="")
    args = parser0.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )