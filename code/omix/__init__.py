__version__ = "0.2.4"
import logging
import sys

logger = logging.getLogger("omix")
# check if logger has been initialized
if not logger.hasHandlers() or len(logger.handlers) == 0:
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
from . import model, tokenizer, utils
from .data_sampler import (DistributedSubsetsBatchSampler, SubsetsBatchSampler,
                           SubsetSequentialSampler)
from .trainer_omix_t import (define_wandb_metrcis, eval_testdata, evaluate,
                      test, train)
