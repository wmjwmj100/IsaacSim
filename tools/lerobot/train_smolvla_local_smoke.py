#!/usr/bin/env python3

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.processor.rename_processor import rename_stats
from lerobot.scripts import lerobot_train as train_module
from lerobot.utils.import_utils import register_third_party_plugins

register_third_party_plugins()

RENAME_MAP = {
    "observation.images.up": "observation.images.camera1",
    "observation.images.side": "observation.images.camera3",
}
SMOLVLA_VLM_MODEL_PATH = os.getenv("SMOLVLA_VLM_MODEL_PATH")

original_make_dataset = train_module.make_dataset
original_make_pre_post_processors = train_module.make_pre_post_processors


def wrapped_make_dataset(train_cfg):
    dataset = original_make_dataset(train_cfg)
    dataset.meta.stats = rename_stats(dataset.meta.stats, RENAME_MAP)
    return dataset


train_module.make_dataset = wrapped_make_dataset


def wrapped_make_pre_post_processors(*args, **kwargs):
    if SMOLVLA_VLM_MODEL_PATH:
        preprocessor_overrides = dict(kwargs.get("preprocessor_overrides") or {})
        preprocessor_overrides["tokenizer_processor"] = {
            **preprocessor_overrides.get("tokenizer_processor", {}),
            "tokenizer_name": SMOLVLA_VLM_MODEL_PATH,
        }
        kwargs["preprocessor_overrides"] = preprocessor_overrides
    return original_make_pre_post_processors(*args, **kwargs)


train_module.make_pre_post_processors = wrapped_make_pre_post_processors


@parser.wrap()
def main(cfg: TrainPipelineConfig):
    train_module.train(cfg)


if __name__ == "__main__":
    main()
