import sys
import ipdb
from glob import glob
import auxiliary.argument_parser as argument_parser
import auxiliary.my_utils as my_utils
import time
import torch
from auxiliary.my_utils import yellow_print
from tqdm import tqdm
import pickle
import numpy as np

"""
Main training script.
author : Thibault Groueix 01.11.2019
"""

opt = argument_parser.parser()
torch.cuda.set_device(opt.multi_gpu[0])
my_utils.plant_seeds(randomized_seed=opt.randomize)
import training.trainer as trainer

trainer = trainer.Trainer(opt)
trainer.build_dataset()
trainer.build_network()
trainer.build_optimizer()
trainer.build_losses()
trainer.start_train_time = time.time()

if opt.demo:
    print('Starting demo')
    # decode
    ipdb.set_trace()
    car_codes = pickle.load(open('car_codes.p', 'rb'))
    mean_code = np.mean(car_codes, axis=1)
    mean_code = np.expand_dims(mean_code, 0)
    trainer.decode(mean_code, 'mean_code.ply')

    # # encode
    # with torch.no_grad():
    #     # trainer.demo(opt.demo_input_path)
    #     fpaths = sorted(glob('dataset/data/ShapeNetV1Renderings/02958343/*/rendering/*.png'))
    #     latent_codes = []
    #     for fpath in tqdm(fpaths):
    #         latent_codes.append(trainer.get_latent_code(fpath))
    #     import ipdb; ipdb.set_trace()
    #     codes = torch.cat(latent_codes, dim=0)
    #     codes = codes.cpu().numpy()
    #     pickle.dump(codes, open('car_codes.p', 'wb'))
    # sys.exit(0)

if opt.run_single_eval:
    with torch.no_grad():
        trainer.test_epoch()
    sys.exit(0)

for epoch in range(trainer.epoch, opt.nepoch):
    trainer.train_epoch()
    with torch.no_grad():
        trainer.test_epoch()
    trainer.dump_stats()
    trainer.increment_epoch()
    trainer.save_network()

yellow_print(f"Visdom url http://localhost:{trainer.opt.visdom_port}/")
yellow_print(f"Netvision report url http://localhost:{trainer.opt.http_port}/{trainer.opt.dir_name}/index.html")
yellow_print(f"Training time {(time.time() - trainer.start_time)//60} minutes.")
