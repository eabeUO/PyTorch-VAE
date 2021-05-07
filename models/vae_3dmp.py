import torch
import math
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from models.types_ import *



class VAE3dmp(BaseVAE):
    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 depth_dim: int,
                 kernels: List = None,
                 mpkernels: List = None,
                 xystrides: List = None,
                 tstrides: List = None,
                 input_size: List = None,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VAE3dmp, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.depth_dim = depth_dim
        
        ##### set default input size if not given. #####
        if input_size is None:
            self.input_size = [in_channels,depth_dim,64,64]
        else:
            self.input_size = input_size

        ##### set default hidden dimensions if not given. #####
        if hidden_dims is None:
            hidden_dims = [16, 32, 64, 128, 256]
            self.hidden_dims = hidden_dims.copy()
        else:
            self.hidden_dims = hidden_dims.copy()

        ##### set default kernels size if not given. #####
        if kernels is None:
            self.kernels = [5 for _ in range(len(hidden_dims))]
        else:
            self.kernels = kernels

        ##### set default kernels size if not given. #####
        if mpkernels is None:
            self.mpkernels = [5 for _ in range(len(hidden_dims))]
        else:
            self.mpkernels = mpkernels

        ##### set default xystride size if not given. #####
        if xystrides is None:
            self.xystrides = [2 for _ in range(len(hidden_dims))]
        else:
            self.xystrides = xystrides

        ##### set default tstride size if not given. #####
        if tstrides is None:
            self.tstrides = [2 for _ in range(len(hidden_dims))]
        else:
            self.tstrides = tstrides

        ##### Check encoder shapes for conv and max pool layers #####
        inshape = self.input_size.copy()
        self.encoder_shapes = []
        for i in range(len(hidden_dims)):
            outshape1 = self.calc_out_dims(inshape,self.hidden_dims[i],ksize=self.kernels[i],stride=1,padding=2)
            outshape = self.calc_out_dims(outshape1,self.hidden_dims[i],ksize=self.mpkernels[i],stride=(self.tstrides[i],self.xystrides[i],self.xystrides[i]),padding=0)
            inshape = outshape
            self.encoder_shapes.append(outshape)
        
        assert self.encoder_shapes[-1] == [hidden_dims[-1],1,4,4], f'Encoder Dims do not match! {self.encoder_shapes}'

        ##### Check decoder shapes for convtranspsoe and upsample layers #####
        self.decoder_shapes = []
        hidden_dims2 = hidden_dims.copy()
        hidden_dims2.reverse()
        outshape1 = self.encoder_shapes[-1]
        for i in range(len(hidden_dims2)):
            outshape = (outshape1[0],outshape1[1]*self.tstrides[i],outshape1[2]*self.xystrides[i],outshape1[3]*self.xystrides[i])
            if i < len(hidden_dims2)-1:
                outshape1 = self.calc_out_dims_trans(outshape,hidden_dims2[i],ksize=self.kernels[i],stride=1,padding=2)
            else:
                outshape1 = self.calc_out_dims_trans(outshape,hidden_dims2[i],ksize=self.kernels[i],stride=1,padding=2)
                outshape1 = self.calc_out_dims_trans(outshape1,1,ksize=self.kernels[i],stride=1,padding=2)
            self.decoder_shapes.append(outshape1)
        
        assert self.decoder_shapes[-1] == self.input_size, f'Decoder Dims do not match! {self.decoder_shapes}'


        # Build Encoder
        self.encoder = nn.ModuleList()
        for layer_n, h_dim in enumerate(hidden_dims):
            self.encoder.add_module(str('conv%i' % layer_n), 
                        nn.Conv3d(in_channels, out_channels=h_dim, 
                                kernel_size=self.kernels[layer_n], 
                                stride=1, #(self.tstrides[layer_n], self.xystrides[layer_n], self.xystrides[layer_n]), 
                                padding=2))
            self.encoder.add_module(str('batchnorm%i' % layer_n), 
                                    nn.BatchNorm3d(h_dim))
            self.encoder.add_module(str('maxpool%i' % layer_n), 
                                        nn.MaxPool3d(kernel_size=self.mpkernels[layer_n], 
                                        stride=(self.tstrides[layer_n], self.xystrides[layer_n], self.xystrides[layer_n]), 
                                        padding=0,return_indices=True))
            self.encoder.add_module(str('relu%i' % layer_n), 
                                        nn.LeakyReLU(0.05))

            in_channels = h_dim

        self.fc_mu = nn.Linear(hidden_dims[-1]*4*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4*4, latent_dim)


        # Build Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*self.encoder_shapes[-1][-1]*self.encoder_shapes[-1][-2]*self.encoder_shapes[-1][-3])
        # self.upsampler = nn.Upsample(scale_factor=2, mode='nearest')

        hidden_dims.reverse()
        self.decoder = nn.ModuleList()
        for layer_n, i in enumerate(range(len(hidden_dims) - 1)):
            # self.decoder.add_module(str('maxunpool%i' % layer_n),
            #                         nn.MaxUnpool3d(kernel_size=2, #self.kernels[layer_n], 
            #                                     stride=(self.tstrides[layer_n], self.xystrides[layer_n], self.xystrides[layer_n]), 
            #                                     padding=0))
            self.decoder.add_module(str('upsample%i' % layer_n),nn.Upsample(scale_factor=(self.tstrides[layer_n], self.xystrides[layer_n], self.xystrides[layer_n]), mode='nearest'))

            self.decoder.add_module(str('convtranspose%i' % layer_n),
                                    nn.ConvTranspose3d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=self.kernels[layer_n],
                                       stride = 1, #(self.tstrides[layer_n], self.xystrides[layer_n], self.xystrides[layer_n]),
                                       padding=2))
            self.decoder.add_module(str('batchnorm%i' % layer_n),
                                    nn.BatchNorm3d(hidden_dims[i + 1]))
            self.decoder.add_module(str('relu%i' % layer_n), nn.LeakyReLU(0.05))

        
        self.final_layer = nn.ModuleList()
        # self.final_layer.add_module(str('last_maxunpool%i' % 0),
        #                         nn.MaxUnpool3d(kernel_size=2, #self.kernels[layer_n], 
        #                                         stride=(self.tstrides[0], self.xystrides[0], self.xystrides[0]), 
        #                                         padding=0))
        self.final_layer.add_module(str('last_upsample%i' % 0),nn.Upsample(scale_factor=(self.tstrides[-1], self.xystrides[-1], self.xystrides[-1]), mode='nearest'))
        self.final_layer.add_module(str('last_convtranspose%i' % 0),
                                nn.ConvTranspose3d(hidden_dims[-1],
                                    hidden_dims[-1],
                                    kernel_size=self.kernels[0],
                                    stride = 1, #(self.tstrides[layer_n], self.xystrides[layer_n], self.xystrides[layer_n]),
                                    padding=2))
        self.final_layer.add_module(str('last_batchnorm%i' % layer_n),
                                    nn.BatchNorm3d(hidden_dims[i + 1]))
        self.final_layer.add_module(str('last_relu%i' % 0), nn.LeakyReLU(0.05)) 
        self.final_layer.add_module(str('last_conv%i' % 0),
                                    nn.Conv3d(hidden_dims[-1], out_channels=self.in_channels,
                                    kernel_size= self.kernels[0], padding=2))
        self.final_layer.add_module(str('last_Tanh%i' % 0), nn.Tanh()) 

    def calc_out_dims(self,inshape,Cout,dialation=1,ksize=1,stride=1,padding=1):
        if type(ksize) == int:
            ksize = (ksize,ksize,ksize)
        if type(dialation) == int:
            dialation = (dialation,dialation,dialation)
        if type(stride) == int:
            stride = (stride,stride,stride)
        if type(padding) == int:
            padding = (padding,padding,padding)

        C,D,H,W = inshape
        Dout = int(((D + 2*padding[0] - dialation[0]*(ksize[0]-1)-1)/stride[0]) + 1)
        Hout = int(((H + 2*padding[1] - dialation[1]*(ksize[1]-1)-1)/stride[1]) + 1)
        Wout = int(((W + 2*padding[2] - dialation[2]*(ksize[2]-1)-1)/stride[2]) + 1)
        return [Cout,Dout,Hout,Wout]

    def calc_out_dims_trans(self,inshape,Cout,dialation=1,ksize=1,stride=1,padding=1,output_padding=0):
        if type(ksize) == int:
            ksize = (ksize,ksize,ksize)
        if type(dialation) == int:
            dialation = (dialation,dialation,dialation)
        if type(stride) == int:
            stride = (stride,stride,stride)
        if type(padding) == int:
            padding = (padding,padding,padding)
        if type(output_padding) == int:
            output_padding = (output_padding,output_padding,output_padding)

        C,D,H,W = inshape
        Dout = int((((D - 1)*stride[0] - 2*padding[0] + dialation[0]*(ksize[0]-1) + output_padding[0])) + 1)
        Hout = int((((H - 1)*stride[1] - 2*padding[1] + dialation[1]*(ksize[1]-1) + output_padding[1])) + 1)
        Wout = int((((W - 1)*stride[2] - 2*padding[2] + dialation[2]*(ksize[2]-1) + output_padding[2])) + 1)
        return [Cout,Dout,Hout,Wout]

    def encode(self, x: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x D x H x W]
        :return: (Tensor) List of latent codes
        """
        # loop over layers, have to collect pool_idx and output sizes if using max pooling to use
        # in unpooling
        pool_idx = []
        target_output_size = []
        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool3d):
                target_output_size.append(x.size())
                x, idx = layer(x)
                pool_idx.append(idx)
            else:
                x = layer(x.contiguous())

        x = torch.flatten(x, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return [mu, log_var, pool_idx, target_output_size]

    def decode(self, pool_idx, target_output_size, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder_input(z)
        result = result.view(-1,self.encoder_shapes[-1][0],self.encoder_shapes[-1][1],self.encoder_shapes[-1][2],self.encoder_shapes[-1][3])
        for name, layer in self.decoder.named_children():
            if isinstance(layer, nn.MaxUnpool3d):
                idx = pool_idx.pop(-1)
                outsize = target_output_size.pop(-1)
                result = layer(result, idx, outsize)
            else:
                result = layer(result)

        for name, layer in self.final_layer.named_children():
            if isinstance(layer, nn.MaxUnpool3d):
                idx = pool_idx.pop(-1)
                outsize = target_output_size.pop(-1)
                result = layer(result, idx, outsize)
            else:
                result = layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, inputs: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var, pool_idx, target_output_size = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(pool_idx, target_output_size, z), inputs, mu, log_var]

    def grab_latents(self, inputs: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var, pool_idx, target_output_size = self.encode(inputs)
        pool_idx2, target_output_size2 = pool_idx.copy(), target_output_size.copy()
        z = self.reparameterize(mu, log_var)
        return  [z, self.generate_from_latents(z), inputs.detach(), mu.detach(), log_var.detach()]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        inputs = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, inputs)  


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0) # orig = .5

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}
        
    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], self.depth_dim//(2**len(self.hidden_dims)), 4, 4)
        for name, layer in self.decoder.named_children():
            if isinstance(layer, nn.MaxUnpool3d):
                result = self.upsampler(result)
            else:
                result = layer(result)

        for name, layer in self.final_layer.named_children():
            if isinstance(layer, nn.MaxUnpool3d):
                result = self.upsampler(result)
            else:
                result = layer(result)
        return result

    def generate_from_latents(self, z: Tensor, **kwargs) -> Tensor:
        """
        Given an input latent z, returns the reconstructed image
        :param x: (Tensor) [B x latent_dim]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1,self.encoder_shapes[-1][0],self.encoder_shapes[-1][1],self.encoder_shapes[-1][2],self.encoder_shapes[-1][3])
        for name, layer in self.decoder.named_children():
            result = layer(result)

        for name, layer in self.final_layer.named_children():
                result = layer(result)
        return result