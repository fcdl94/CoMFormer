from detectron2.config import CfgNode as CN


def add_continual_config(cfg):
    cfg.NAME = "Exp"
    cfg.CONT = CN()
    cfg.CONT.BASE_CLS = 15
    cfg.CONT.INC_CLS = 5
    cfg.CONT.ORDER = list(range(1, 21))
    cfg.CONT.TASK = 0
    cfg.CONT.WEIGHTS = None
    cfg.CONT.MODE = "overlap"  # Choices "overlap", "disjoint", "sequential"

    cfg.CONT.DIST = CN()
    cfg.CONT.DIST.KD_WEIGHT = 0.
    cfg.CONT.DIST.ALPHA = 1.
    cfg.CONT.DIST.UCE = False
    cfg.CONT.DIST.UKD = False
    cfg.CONT.DIST.KD_DEEP = True
    cfg.CONT.DIST.USE_NEW = False