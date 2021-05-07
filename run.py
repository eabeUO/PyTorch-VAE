import yaml
import argparse
import numpy as np
import os

from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/WC_vae3dmp.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tt_logger = TestTubeLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    debug=False,
    create_git_tag=False,
)
##### Save parameters fro every experiment #####
savefile = os.path.join(config['logging_params'],'VAE3dmp',f'version_{tt_logger.version}',os.path.basename(args.filename))
with open(savefile,'w') as file: 
    try:
        yaml.dump(config, file)
    except yaml.YAMLError as exc:
        print(exc)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'])

runner = Trainer(weights_save_path=f"{tt_logger.save_dir}",
                 min_epochs=1,
                 logger=tt_logger,
                 log_every_n_steps=100,
                 limit_train_batches=1.,
                 val_check_interval=1.,
                 num_sanity_val_steps=0,
                 log_gpu_memory='min_max',
                 **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment)