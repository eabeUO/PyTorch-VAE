import torch
import math
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from models.types_ import *
from ConvLSTM import ConvLSTM


class VAE_rnn(BaseVAE):
    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 depth_dim: int,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 kernels: List = None,
                 mpkernels: List = None,
                 xystrides: List = None,
                 tstrides: List = None,
                 input_size: List = None,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VAE_rnn, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.depth_dim = depth_dim

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

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
        
        assert self.encoder_shapes[-1] == [hidden_dims[-1],depth_dim,4,4], f'Encoder Dims do not match! {self.encoder_shapes}'

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
        for layer_n, h_dim in enumerate(self.hidden_dims):
            self.encoder.add_module(str('ConvLSTM{}'.format(layer_n)),ConvLSTM(input_dim = in_channels,
                                                                        hidden_dim = [h_dim],
                                                                        kernel_size= (self.kernels[layer_n],self.kernels[layer_n]),
                                                                        num_layers = 1,
                                                                        batch_first=True,
                                                                        use_transpose=False,
                                                                        ))

            self.encoder.add_module(str('batchnorm%i' % layer_n), 
                                        nn.BatchNorm3d(h_dim))
            self.encoder.add_module(str('maxpool%i' % layer_n), 
                                        nn.MaxPool2d(kernel_size=self.mpkernels[layer_n], 
                                        stride=(self.xystrides[layer_n], self.xystrides[layer_n]), 
                                        padding=0,return_indices=False))
            self.encoder.add_module(str('relu%i' % layer_n), 
                                        nn.LeakyReLU(0.05))
            in_channels = h_dim
        self.fc_mu = nn.Linear(self.hidden_dims[-1]*self.depth_dim*4*4, self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1]*self.depth_dim*4*4, self.latent_dim)
        
        # Build Decoder
        self.decoder_input = nn.Linear(self.latent_dim, self.hidden_dims[-1]*self.encoder_shapes[-1][-1]*self.encoder_shapes[-1][-2]*self.encoder_shapes[-1][-3])

        hidden_dims.reverse()
        self.decoder = nn.ModuleList()
        for layer_n, i in enumerate(range(len(hidden_dims) - 1)):
            self.decoder.add_module(str('ConvLSTM{}_transpose'.format(layer_n)), ConvLSTM(input_dim  = hidden_dims[i],
                                                                                    hidden_dim  = hidden_dims[i+1],
                                                                                    kernel_size = (self.kernels[layer_n],self.kernels[layer_n]),
                                                                                    num_layers  = 1,
                                                                                    batch_first = True,
                                                                                    use_transpose = True,
                                                                                ))
            self.decoder.add_module(str('upsample%i' % i),nn.Upsample(scale_factor=(1, self.xystrides[-1], self.xystrides[-1]), mode='nearest'))

            self.decoder.add_module(str('batchnorm%i' % layer_n),
                                    nn.BatchNorm3d(hidden_dims[i + 1]))
            self.decoder.add_module(str('relu%i' % layer_n), nn.LeakyReLU(0.05))


        self.final_layer = nn.ModuleList()
        self.final_layer.add_module(str('ConvLSTM{}_transpose'.format(layer_n)), ConvLSTM(input_dim   = hidden_dims[-1],
                                                                                            hidden_dim  = hidden_dims[-1],
                                                                                            kernel_size = (self.kernels[-1],self.kernels[-1]),
                                                                                            num_layers  = 1,
                                                                                            batch_first = True,
                                                                                            use_transpose = True,
                                                                                        ))
        self.final_layer.add_module(str('upsample%i' % i),nn.Upsample(scale_factor=(1, self.xystrides[-1], self.xystrides[-1]), mode='nearest'))

        self.final_layer.add_module(str('batchnorm%i' % layer_n), nn.BatchNorm3d(hidden_dims[-1]))
        self.final_layer.add_module(str('relu%i' % layer_n), nn.LeakyReLU(0.05))
        self.final_layer.add_module(str('last_conv%i' % 0), nn.Conv3d(hidden_dims[-1], out_channels=self.in_channels,
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
        # Dout = int(((D + 2*padding[0] - dialation[0]*(ksize[0]-1)-1)/stride[0]) + 1)
        Hout = int(((H + 2*padding[1] - dialation[1]*(ksize[1]-1)-1)/stride[1]) + 1)
        Wout = int(((W + 2*padding[2] - dialation[2]*(ksize[2]-1)-1)/stride[2]) + 1)
        return [Cout,D,Hout,Wout]

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
        # Dout = int((((D - 1)*stride[0] - 2*padding[0] + dialation[0]*(ksize[0]-1) + output_padding[0])) + 1)
        Hout = int((((H - 1)*stride[1] - 2*padding[1] + dialation[1]*(ksize[1]-1) + output_padding[1])) + 1)
        Wout = int((((W - 1)*stride[2] - 2*padding[2] + dialation[2]*(ksize[2]-1) + output_padding[2])) + 1)
        return [Cout,D,Hout,Wout]

    def encode(self, x: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x D x H x W]
        :return: (Tensor) List of latent codes
        """
        for name, layer in self.encoder.named_children():
            if isinstance(layer,ConvLSTM):
                if x.shape[1] != self.depth_dim:
                    x = x.permute(0,2,1,3,4)
                else:
                    pass
                x, state = layer(x)
                x = x[0].permute(0,2,1,3,4)
            elif isinstance(layer,nn.MaxPool2d):
                shape = x.shape
                x = x.view(shape[0],shape[1]*shape[2],shape[3],shape[4])
                x = layer(x)
                x = x.view(shape[0],shape[1],shape[2],x.shape[-2],x.shape[-1])
            else:
                x = layer(x)

        x = torch.flatten(x, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder_input(z)

        result = result.view(-1,self.encoder_shapes[-1][0],self.encoder_shapes[-1][1],self.encoder_shapes[-1][-2],self.encoder_shapes[-1][-1])
        result = result.permute(0,2,1,3,4)
        for name, layer in self.decoder.named_children():
            if isinstance(layer,ConvLSTM):
                if result.shape[1] != self.depth_dim:
                    result = result.permute(0,2,1,3,4)
                else:
                    pass
                result, _ = layer(result)
                result = result[0].permute(0,2,1,3,4)
            else:
                result = layer(result)

        for name, layer in self.final_layer.named_children():
            if isinstance(layer,ConvLSTM):
                if result.shape[1] != self.depth_dim:
                    result = result.permute(0,2,1,3,4)
                else:
                    pass
                result, _ = layer(result)
                result = result[0].permute(0,2,1,3,4)
            else:
                result = layer(result)
        if result.shape[1] != self.depth_dim:
            result = result.permute(0,2,1,3,4)
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
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), inputs, mu, log_var]

    def grab_latents(self, inputs: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        return  [z, self.generate_from_latents(z), inputs.detach(), mu.detach(), log_var.detach()]

    # def loss_function(self,
    #                   *args,
    #                   **kwargs) -> dict:
    #     """
    #     Computes the VAE loss function.
    #     KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    #     :param args:
    #     :param kwargs:
    #     :return:
    #     """
    #     recons = args[0]
    #     inputs = args[1]
    #     mu = args[2]
    #     log_var = args[3]

    #     kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
    #     recons_loss =F.mse_loss(recons, inputs)  


    #     kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0) # orig = .5

    #     loss = recons_loss + kld_weight * kld_loss
    #     return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

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

        result = self.decode(z)
        return result

    def generate_from_latents(self, z: Tensor, **kwargs) -> Tensor:
        """
        Given an input latent z, returns the reconstructed image
        :param x: (Tensor) [B x latent_dim]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decode(z)
        return result

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]