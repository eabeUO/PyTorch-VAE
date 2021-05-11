import os
import numpy as np
import torch
from matplotlib import colors
from matplotlib.animation import FFMpegWriter, FuncAnimation
from tqdm import tqdm, trange

from myutils import *


def make_lat_anim(model, rootdir, FigurePath, config, version, dataset, num_samples=200, nstd=30, StartInd=0, device='cuda'):
    print('Starting Latent Traversal')
    ##### Taverse componenets #####
    dtrange = np.floor(nstd*np.std(zt, axis=0))
    save_lats = check_path(rootdir, 'LatentTravs/version_{:d}'.format(version))
    if os.path.exists(os.path.join(save_lats, 'LatTrav_{}_range{:d}.npy'.format(dataset.data_paths['BasePath'][StartInd], int(nstd)))):
        tot_samps = np.load(os.path.join(save_lats, 'LatTrav_{}_range{:d}.npy'.format(
            dataset.data_paths['BasePath'][StartInd], int(nstd))))
    else:
        tot_samps = np.zeros((config['model_params']['latent_dim'], num_samples*2, config['exp_params']
                             ['imgH_size'], config['exp_params']['imgW_size']))  # Comp x Trav x H x W
        # with PdfPages(os.path.join(FigurePath,'Latent_traversal.pdf')) as pdf:
        for comp in trange(config['model_params']['latent_dim']):
            dt = dtrange[comp]/num_samples
            epses = np.round(
                np.linspace(-dtrange[comp], dtrange[comp], num=num_samples*2), decimals=6)
            z_trav = []
            # zt[:1,:]-np.mean(zt,axis=0)# np.random.randn(1,model.latent_dim) #
            z_temp = np.zeros((config['model_params']['latent_dim']))
            for eps in epses:
                tempadd = np.zeros_like(z_temp)
                tempadd[comp] = eps
                z_trav.append(z_temp + tempadd)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    z = torch.Tensor(z_trav).to(device)
                    samples = model.generate_from_latents(z)

            tot_samps[comp] = samples[:, 0, 0].cpu().detach().float().numpy()

            del samples, z
            torch.cuda.empty_cache()
        print('tot_samps: ', tot_samps.shape)

        np.save(os.path.join(save_lats, 'LatTrav_{}_range{:d}.npy'.format(
            dataset.data_paths['BasePath'][StartInd], int(nstd))), tot_samps)

    def init():
        for n in range(config['model_params']['latent_dim']):
            axs[n].axis('off')
        plt.tight_layout()

    def update(t):
        for n in range(config['model_params']['latent_dim']):
            ims[n].set_data(tot_samps[n, t])
        plt.draw()

    print('Creating Latent Animation')
    t = 0
    x, y = [], []
    lat_dims = config['model_params']['latent_dim']
    fig, axs = plt.subplots(int(np.round(np.sqrt(lat_dims))), int(
        np.ceil(np.sqrt(lat_dims))), figsize=(15, 16))
    axs = axs.flatten()
    ims = []
    for n in range(config['model_params']['latent_dim']):
        ims.append(axs[n].imshow(tot_samps[n, t],
                   cmap='gray', norm=colors.Normalize()))
        axs[n].axis('off')
        axs[n].set_title('{:d}'.format(n))
    plt.tight_layout()
    ani = FuncAnimation(fig, update, range(tot_samps.shape[1]), init_func=init)
    vpath = check_path(FigurePath, 'version_{:d}'.format(version))
    writervideo = FFMpegWriter(fps=60)
    print('Saving Latent Animation')
    ani.save(os.path.join(vpath, 'LatTrav_{}_range{:d}.mp4'.format(dataset.data_paths['BasePath'][StartInd], int(nstd))), writer=writervideo,
             progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'))
    print('DONE!!!')
