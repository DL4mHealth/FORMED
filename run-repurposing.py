import os
import random
import sys

import numpy as np
import torch
from rich import print

sys.path.append("./src")

from src.experiment.classification_repurposing import (
    ExpConfig,
    ExpRepurposing,
)

if __name__ == "__main__":
    exp_config = ExpConfig()
    parser = exp_config.get_argparser()
    args = parser.parse_args()
    exp_config.update(args)
    exp_config.use_gpu = exp_config.use_gpu and torch.cuda.is_available()
    if exp_config.use_gpu and exp_config.use_multi_gpu:
        exp_config.devices = exp_config.devices.strip().replace(" ", "")
        exp_config.gpu = int(exp_config.devices.split(",")[0])

    print("Experiment settings:")
    print(exp_config)

    if exp_config.is_train:
        for ii in range(exp_config.num_itr):
            print(
                ">>>>>>> Initializing experiment environment >>>>>>>>>>>>>>>>>>>>>>>>>>"
            )
            _seed = exp_config.seed + ii
            os.environ["PYTHONHASHSEED"] = str(_seed)
            random.seed(_seed)
            np.random.seed(_seed)
            torch.manual_seed(_seed)
            torch.cuda.manual_seed(_seed)
            torch.cuda.manual_seed_all(_seed)

            setting = str(exp_config) + f"-sd_{_seed}"
            exp = ExpRepurposing(exp_config)
            print(f">>>>>>> start training : {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>")
            exp.train(setting)

            print(f">>>>>>> start testing : {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>>")
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        for ii in range(exp_config.num_itr):
            _seed = exp_config.seed + ii
            os.environ["PYTHONHASHSEED"] = str(_seed)
            random.seed(_seed)
            np.random.seed(_seed)
            torch.manual_seed(_seed)
            torch.cuda.manual_seed(_seed)
            torch.cuda.manual_seed_all(_seed)

            setting = str(exp_config) + f"-sd_{_seed}"
            exp = ExpRepurposing(exp_config)
            print(f">>>>>>> start testing : {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>>")
            exp.test(setting)
            torch.cuda.empty_cache()
