import glob
import os
import yaml
import argparse
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

from functools import partial
from tqdm import trange, tqdm
# import umap
# import umap.plot 

import torch
import torchvision
import torch.nn as nn
# import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from datasets import WCDataset, WCShotgunDataset, WC3dDataset
from myutils import *
from models.vanilla_vae import VanillaVAE
from models.vae_3dmp import VAE3dmp
from models.vae_3d import VAE3d
from experiment import VAEXperiment


import plotly.express as px
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib import colors

rootdir = os.path.expanduser('~/Research/FMEphys/')

# Set up partial functions for directory managing
join = partial(os.path.join,rootdir)
checkDir = partial(check_path,rootdir)
FigurePath = checkDir('Figures')
savefigs=False

def arg_parser(): 
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--BatchSize', type=int, default=120)
    parser.add_argument('--StartInd', type=int, default=0)
    parser.add_argument('--NumBatches', type=int, default=1786) # 1786 
    parser.add_argument('--source_path', type=str,
                        default='~/Research/FMEphys/',
                        help='Path to load files from')
    parser.add_argument('--modeltype', type=str, default='3dmp')
    parser.add_argument('--use_subset', type=bool, default=False)

    parser.add_argument('--savedata', type=bool, default=False)
    parser.add_argument('--savefigs', type=bool, default=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parser()


    ##### Loading in trained Network #####
    n = -1
    version = 4
    modeltype = '3dmp' # '3d'
    if modeltype=='shotgun': 
        filename =  os.path.join(os.path.expanduser('~/Research/Github/'),'PyTorch-VAE','configs/WC_vae_shotgun.yaml')
        ckpt_path = glob.glob(os.path.expanduser('~/Research/FMEphys/logs2/VanillaVAE/version_3/checkpoints/*.ckpt'))[n]
    elif modeltype=='vanilla':
        filename =  os.path.join(os.path.expanduser('~/Research/Github/'),'PyTorch-VAE','configs/WC_vae.yaml')
        ckpt_path = glob.glob(os.path.expanduser('~/Research/FMEphys/logs2/VanillaVAE/version_0/checkpoints/*.ckpt'))[n]
    elif modeltype=='3d':
        filename =  os.path.join(os.path.expanduser('~/Research/Github/'),'PyTorch-VAE','configs/WC_vae3d.yaml')
        ckpt_path = glob.glob(os.path.expanduser('~/Research/FMEphys/logs2/VAE3d/version_4/checkpoints/*.ckpt'))[n]
    elif modeltype=='3dmp':
        filename =  os.path.join(os.path.expanduser('~/Research/FMEphys/logs2/VAE3dmp/version_{:d}/WC_vae3dmp.yaml'.format(version)))
        ckpt_path = glob.glob(os.path.expanduser('~/Research/FMEphys/logs2/VAE3dmp/version_{:d}/checkpoints/*.ckpt'.format(version)))[n]
    else:
        raise ValueError(f'{n} is not a valid model type')
    print(ckpt_path)

    with open(filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    if modeltype=='shotgun': 
        config['exp_params']['data_path'] = os.path.expanduser('~/Research/FMEphys/')
        config['exp_params']['csv_path_train'] = os.path.expanduser('~/Research/FMEphys/WCShotgun_Train_Data.csv')
        config['exp_params']['csv_path_val'] = os.path.expanduser('~/Research/FMEphys/WCShotgun_Val_Data.csv')
        config['logging_params']['save_dir'] = os.path.expanduser('~/Research/FMEphys/logs2/')
    elif modeltype=='vanilla':
        config['exp_params']['data_path'] = os.path.expanduser('~/Research/FMEphys/')
        config['exp_params']['csv_path_train'] = os.path.expanduser('~/Research/FMEphys/WC_Train_Data.csv')
        config['exp_params']['csv_path_val'] = os.path.expanduser('~/Research/FMEphys/WC_Val_Data.csv')
        config['logging_params']['save_dir'] = os.path.expanduser('~/Research/FMEphys/logs2/')
    elif (modeltype=='3d') | (modeltype=='3dmp'):
        config['exp_params']['data_path'] = os.path.expanduser('~/Research/FMEphys/data')
        config['exp_params']['csv_path_train'] = os.path.expanduser('~/Research/FMEphys/WC3d_Train_Data_SingVid.csv')
        config['exp_params']['csv_path_val'] = os.path.expanduser('~/Research/FMEphys/WC3d_Val_Data_SingVid.csv')
        config['logging_params']['save_dir'] = os.path.expanduser('~/Research/FMEphys/logs2/')
    
    print(config)

    if modeltype=='3d':
        model = VAE3d(**config['model_params'])
    if modeltype=='3dmp':
        model = VAE3dmp(**config['model_params'])
    else:
        model = VanillaVAE(**config['model_params'])
    experiment = VAEXperiment(model,config['exp_params'])

    experiment = experiment.load_from_checkpoint(ckpt_path,vae_model=model,params=config['exp_params'])
    experiment.to(device)
    print('Loaded')

    ##### Initialize Datasets and DataLoaders #####
    StartInd = args.StartInd
    NumBatches= args.NumBatches
    config['exp_params']['batch_size'] = args.BatchSize
    if modeltype=='shotgun': 
        dataset = WCShotgunDataset(root_dir = config['exp_params']['data_path'],
                                    csv_file = config['exp_params']['csv_path_train'],
                                    N_fm=config['exp_params']['N_fm'],
                                    transform=experiment.data_transforms())
    elif modeltype=='vanilla':
        dataset = WCDataset(root_dir = config['exp_params']['data_path'],
                            csv_file = config['exp_params']['csv_path_train'],
                            transform=experiment.data_transforms())
    elif (modeltype=='3d') | (modeltype=='3dmp'):
        dataset = WC3dDataset(root_dir = config['exp_params']['data_path'],
                                    csv_file = config['exp_params']['csv_path_train'],
                                    N_fm=config['exp_params']['N_fm'],
                                    transform=experiment.data_transforms())
    if args.use_subset:
        train_dataset = Subset(dataset,torch.arange(StartInd,StartInd+config['exp_params']['batch_size']*NumBatches)) # 107162
    else:
        train_dataset = dataset
    train_dataloader = DataLoader(train_dataset,
                                batch_size= config['exp_params']['batch_size'],
                                shuffle = False,
                                drop_last=True,
                                num_workers=7,
                                pin_memory=False,
                                prefetch_factor=10)

    postdir = check_path(rootdir,'post_data/version_{:d}'.format(version))
    if (os.path.exists(os.path.join(postdir,'zt_{}.npy'.format(dataset.data_paths['BasePath'][StartInd]))) == False) | (args.savedata==True):
        print('Grabbing Latents')
        ##### Grab latents, reconstructions and frames #####
        zt = np.empty((len(train_dataloader),config['exp_params']['batch_size'],config['model_params']['latent_dim']))
        # recont = np.empty((NumBatches,config['exp_params']['batch_size'],config['exp_params']['N_fm'],config['exp_params']['imgH_size'],config['exp_params']['imgW_size']), dtype=np.float32)
        # batcht = np.empty((NumBatches,config['exp_params']['batch_size'],config['exp_params']['N_fm'],config['exp_params']['imgH_size'],config['exp_params']['imgW_size']), dtype=np.float32)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for n, batch in enumerate(tqdm(train_dataloader)):
                    z,recon,inputs,_,_ = model.grab_latents(batch.to(device))
                    zt[n] = z.cpu().numpy()
                    # recont[n] = recon[:,0].cpu().numpy()
                    # batcht[n] = inputs[:,0].cpu().numpy()
        del z, recon, inputs
        torch.cuda.empty_cache()
        zt = zt.reshape(-1,config['model_params']['latent_dim'])
        # recont = recont.reshape(-1,config['exp_params']['N_fm'],config['exp_params']['imgH_size'],config['exp_params']['imgW_size'])
        # batcht = batcht.reshape(-1,config['exp_params']['N_fm'],config['exp_params']['imgH_size'],config['exp_params']['imgW_size'])
        np.save(os.path.join(postdir,'zt_{}.npy'.format(dataset.data_paths['BasePath'][StartInd])),zt)

    else: 
        print('Loading Latents')
        zt = np.load(os.path.join(postdir,'zt_{}.npy'.format(dataset.data_paths['BasePath'][StartInd])))



    ########## Load Ephys data ##########
    ##### Load Ephys, Tstamps #####
    print('Loading Ephys')
    nframes = zt.shape[0] #len(train_dataloader)*args.BatchSize
    ephys_path = glob.glob(os.path.join(rootdir,'data',dataset.data_paths['BasePath'][StartInd][:-5] + 'ephys','*merge.json'))[0]
    TS_path = glob.glob(os.path.join(rootdir,'data',dataset.data_paths['BasePath'][StartInd][:-5] + 'ephys','*TSformatted.csv'))[0]
    ephys_df = pd.read_json(ephys_path)
    worldT = pd.read_csv(TS_path)['0']
    worldT = worldT[StartInd:nframes] - ephys_df['t0'][0]
    if worldT[0]<-600:
        worldT = worldT + 8*60*60

    good_cells = ephys_df[ephys_df['group']=='good']
    n_units = len(good_cells)
    # batch = next(iter(train_dataloader))
    # world_cam = batch[:,0].cpu().numpy().copy()
    # world_norm = normimgs(world_cam)

    model_dt = 0.025
    model_t = np.arange(0,np.max(worldT),model_dt)
    model_nsp = np.zeros((len(good_cells),len(model_t)))

    # get spikes / rate
    bins = np.append(model_t,model_t[-1]+model_dt)
    for i,ind in enumerate(good_cells.index):
        model_nsp[i,:],bins = np.histogram(good_cells.at[ind,'spikeT'],bins)

    # Set up interp for latents
    latInterp = interp1d(worldT,zt,axis=0, kind='nearest', bounds_error = False)

    nks = config['model_params']['input_size'][1:]; nk = nks[0]*nks[1];    
    model_lat = np.zeros((len(model_t),config['model_params']['latent_dim']))
    for i in trange(len(model_t)):
        model_lat[i] = latInterp(model_t[i] + model_dt/2)
    model_lat[np.isnan(model_lat)]=0

    ##### Calculate STA #####
    lagRange = np.concatenate(([-30],np.arange(-2,8,2)))
    stalat = np.zeros((n_units,len(lagRange),config['model_params']['latent_dim']))
    for c, ind in enumerate(good_cells.index):
        for  lagInd, lag in enumerate(lagRange):
            sp = model_nsp[c,:].copy()
            sp = np.roll(sp,-lag)
            sta = model_lat.T@sp
            stalat[c,lagInd] = sta/np.sum(sp)
        print(f'Cell:{ind}, nspks:{np.sum(sp)}')

    sta_shape = stalat.shape
    stalat = stalat.reshape(-1,sta_shape[-1])
    stalat = stalat - np.mean(zt,axis=0)

    ##### Push through decoder #####
    num_samples = stalat.shape[0]
    sta_z = torch.Tensor(stalat).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            samples = model.generate_from_latents(sta_z)

    ##### Plot STA #####
    fig, axs = plt.subplots(1,figsize=(10,20))
    im_grid = torchvision.utils.make_grid(samples[:,0,:1].cpu().float(),nrow=len(lagRange),normalize=True).cpu()
    axs.imshow(im_grid.permute(1,2,0))
    axs.set_title('Decoded STA')
    # axs.xaxis.set_ticks_position('top')
    axs.set_xticks(np.arange(32,im_grid.shape[-1],65))
    axs.set_xticklabels(lagRange)
    axs.set_xlabel('Lag')
    # axs.xaxis.set_label_position('top') # <-- This doesn't work!
    axs.set_yticks(np.arange(32,im_grid.shape[-2],66))
    axs.set_yticklabels(good_cells.index)
    axs.set_ylabel('Unit #')
    
    if args.savefigs:
        fig.savefig(os.path.join(check_path(FigurePath,'version_{:d}'.format(version)),'STA_Model{}.png'.format(config['exp_params']['imgH_size'])))
    
    del samples, sta_z
    torch.cuda.empty_cache()

    print('Starting Latent Traversal')
    ##### Taverse componenets #####
    num_samples=200
    nstd = 30
    dtrange = np.floor(nstd*np.std(zt,axis=0))
    save_lats = check_path(rootdir,'LatentTravs/version_{:d}'.format(version))
    if os.path.exists(os.path.join(save_lats,'LatTrav_{}_range{:d}.npy'.format(dataset.data_paths['BasePath'][StartInd],int(nstd)))):
        tot_samps = np.load(os.path.join(save_lats,'LatTrav_{}_range{:d}.npy'.format(dataset.data_paths['BasePath'][StartInd],int(nstd))))
    else:
        tot_samps = np.zeros((config['model_params']['latent_dim'],num_samples*2,config['exp_params']['imgH_size'],config['exp_params']['imgW_size'])) # Comp x Trav x H x W
        # with PdfPages(os.path.join(FigurePath,'Latent_traversal.pdf')) as pdf:
        for comp in trange(config['model_params']['latent_dim']):
            dt = dtrange[comp]/num_samples
            epses = np.round(np.linspace(-dtrange[comp],dtrange[comp],num=num_samples*2),decimals=6)
            z_trav = [] 
            z_temp = np.zeros((config['model_params']['latent_dim']))#zt[:1,:]-np.mean(zt,axis=0)# np.random.randn(1,model.latent_dim) # 
            for eps in epses:
                tempadd = np.zeros_like(z_temp)
                tempadd[comp] = eps
                z_trav.append(z_temp + tempadd)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    z = torch.Tensor(z_trav).to(device)
                    samples = model.generate_from_latents(z)

            tot_samps[comp] = samples[:,0,0].cpu().detach().float().numpy()

            del samples,z
            torch.cuda.empty_cache()
        print('tot_samps: ',tot_samps.shape)
    

        np.save(os.path.join(save_lats,'LatTrav_{}_range{:d}.npy'.format(dataset.data_paths['BasePath'][StartInd],int(nstd))),tot_samps)

    def init():
        for n in range(config['model_params']['latent_dim']):
            axs[n].axis('off')
        plt.tight_layout()

    def update(t):
        for n in range(config['model_params']['latent_dim']):
            ims[n].set_data(tot_samps[n,t])
        plt.draw()
    
    print('Creating Latent Animation')
    t = 0
    x,y = [],[]
    lat_dims=config['model_params']['latent_dim']
    fig, axs = plt.subplots(int(np.round(np.sqrt(lat_dims))),int(np.ceil(np.sqrt(lat_dims))),figsize=(15,16))  
    axs = axs.flatten()
    ims = []
    for n in range(config['model_params']['latent_dim']):
        ims.append(axs[n].imshow(tot_samps[n,t],cmap='gray',norm=colors.Normalize()))
        axs[n].axis('off')
        axs[n].set_title('{:d}'.format(n))
    plt.tight_layout()
    ani = FuncAnimation(fig, update, range(tot_samps.shape[1]), init_func=init)  
    vpath = check_path(FigurePath,'version_{:d}'.format(version))
    writervideo = FFMpegWriter(fps=60) 
    print('Saving Latent Animation')
    ani.save(os.path.join(vpath,'LatTrav_{}_range{:d}.mp4'.format(dataset.data_paths['BasePath'][StartInd],int(nstd))), writer=writervideo,
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}')) 
    print('DONE!!!')