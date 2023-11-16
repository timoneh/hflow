from functools import reduce
from itertools import accumulate

import numpy as np
import torch



def bil():
    return 1.0/8.0*np.array([1,3,3,1])

# Will not get edges exactly correct when compared with PIL's resizing due to kernel weights not being set to zero when outside the image.
def decimate2_torch(x, k):
    k = k.expand(x.shape[1],-1,-1,-1)
    n = 1 if k.shape[-1] == 4 else 3
    x = torch.nn.functional.pad(x, (n,n,n,n), "reflect")
    return torch.nn.functional.conv2d(x, k, stride=2, groups=x.shape[1])


def interpolate2_torch(x, k):
    k = k.expand(x.shape[1],-1,-1,-1)
    t = torch.zeros(size=(x.shape[0], x.shape[1], 2*x.shape[2], 2*x.shape[3]), device=x.device)
    t[:,:,1::2,1::2] = x 
    t = torch.nn.functional.pad(t, (1,2,1,2), "reflect")
    x_conv = torch.nn.functional.conv2d(t, 4*k, stride=1, groups=t.shape[1])
    return x_conv


class PoolingCouplingCNN(torch.nn.Module):

    def __init__(self, c_in, c_out, c_middle, res, pooling_type="unet", ind=0, isCond=False,
                preproc_out_c=3, kernel_size=3, unet_n_downscales=2, parent_ds_us=None, unet_c_mult=2.0,
                masking_type="spatial"):
        super(PoolingCouplingCNN, self).__init__()

        k_size = kernel_size
        k_size_unet = kernel_size
        if "reduced" in pooling_type:
            k_size_unet = 1

        cond_base = unet_c_mult
        self.ind = ind
        self.scaling = 1.0

        self.lrelu_slope = 0.1
        self.us = None if parent_ds_us is None else parent_ds_us[1]
        self.ds = None if parent_ds_us is None else parent_ds_us[0]


        self.n_downscale_per_layer = 1
        if "double" in pooling_type:
            self.n_downscale_per_layer = 2
        if "third-c" in pooling_type:
            c_middle = c_middle // 3


        self.c_in = c_in
        self.c_base = c_middle
        self.res = res
        self.isCond = isCond
        self.res_lowest = self.res//(2**unet_n_downscales)

        self.masking_type = masking_type


        self.mask_in = torch.ones(size=(1, self.c_in, self.res, self.res))

        if masking_type == "channels" and self.c_in == 1:
            masking_type = "blocks"
            print("Changed masking type of PoolingCouplingCNN to --blocks-- from --channels-- since there is only ONE channel at res {}".format(self.res))

        if masking_type == "channels":
            out_channels = int(np.ceil((self.c_in*0.5)))
            in_channels = self.c_in - out_channels
            self.mask_in[:,in_channels:,...] = 0.0 #see only first channels
        elif masking_type == "vhalfs":
            self.mask_in[:,:,:,0:self.res//2] = 0.0
        elif masking_type == "hhalfs":
            self.mask_in[:,:,0:self.res//2,:] = 0.0
        elif masking_type == "spatial":
            self.mask_in[:,:,0::2, 0::2] = 0.0
            self.mask_in[:,:,1::2, 1::2] = 0.0
        elif masking_type == "blocks":
            m_ones = torch.ones(size=(1, self.c_in, 2, 2))
            m_zeros = torch.zeros(size=(1, self.c_in, 2, 2))
            m = torch.cat([torch.cat([m_ones, m_zeros], dim=-1), torch.cat([m_zeros, m_ones], dim=-1)], dim=-2)
            R = self.res//m.shape[-1]
            if R < 1:
                raise ValueError("Checkerboard-pattern needs at least resolution 4, is {}".format(self.res))
            self.mask_in = m.repeat(1,1,R,R)
        elif masking_type == "lblocks":
            m_ones = torch.ones(size=(1, self.c_in, 4, 4))
            m_zeros = torch.zeros(size=(1, self.c_in, 4, 4))
            m = torch.cat([torch.cat([m_ones, m_zeros], dim=-1), torch.cat([m_zeros, m_ones], dim=-1)], dim=-2)
            R = self.res//m.shape[-1]
            if R < 1:
                raise ValueError("Checkerboard-pattern needs at least resolution 8, is {}".format(self.res))
            self.mask_in = m.repeat(1,1,R,R)
        elif masking_type == "hblocks":
            m_ones = torch.ones(size=(1, self.c_in, 8, 8))
            m_zeros = torch.zeros(size=(1, self.c_in, 8, 8))
            m = torch.cat([torch.cat([m_ones, m_zeros], dim=-1), torch.cat([m_zeros, m_ones], dim=-1)], dim=-2)
            R = self.res//m.shape[-1]
            if R < 1:
                raise ValueError("Checkerboard-pattern needs at least resolution 16, is {}".format(self.res))
            self.mask_in = m.repeat(1,1,R,R)
        else:
            raise NotImplementedError("Masking type: {}".format(self.masking_type))

        self.mask_out = torch.where(self.mask_in>0.0, torch.zeros_like(self.mask_in), torch.ones_like(self.mask_in))
        self.mask_in = torch.nn.Parameter(self.mask_in.float(), requires_grad=False)
        self.mask_out = torch.nn.Parameter(self.mask_out.float(), requires_grad=False)



        if masking_type != "channels" and (ind%2)==0:
            self.mask_in, self.mask_out = self.mask_out, self.mask_in


        self.mask_in.data = self.mask_in.data * np.sqrt(2.0)



        if isCond:
            self.conv_cond_connect_pre = MyConv(preproc_out_c, preproc_out_c//2, 1, self.scaling, nonlin="tanh")
            self.conv_cond_connect_un = MyConv(c_in+preproc_out_c//2, self.c_base, 3, self.scaling, nonlin="lrelu")
            self.conv_1 = MyConv(self.c_base, self.c_base, k_size, self.scaling, nonlin="lrelu")
        else:
            self.conv_1 = MyConv(c_in, self.c_base, k_size, self.scaling, nonlin="lrelu")

        self.conv_2 = MyConv(self.c_base, self.c_base, k_size, self.scaling, nonlin="lrelu")
        self.conv_3 = MyConv(self.c_base, self.c_base, k_size, self.scaling, nonlin="lrelu")
        self.n_downscale = unet_n_downscales


        self.down = torch.nn.ModuleList([MyConv(int(self.c_base*(cond_base**i)), int(self.c_base*(cond_base**(i+1))), k_size_unet, self.scaling, nonlin="lrelu") for i in range(self.n_downscale)])
        self.down_b = torch.nn.ModuleList([MyConv(int(self.c_base*(cond_base**(i+1))), int(self.c_base*(cond_base**(i+1))), k_size_unet, self.scaling, nonlin="lrelu") for i in range(self.n_downscale)])
        self.down_c = torch.nn.ModuleList([MyConv(int(self.c_base*(cond_base**i))+int(self.c_base*(cond_base**(i+1))), int(self.c_base*(cond_base**(i+1))), k_size_unet, self.scaling, nonlin="linear") for i in range(self.n_downscale)])

        self.up = torch.nn.ModuleList([MyConv(int(self.c_base*(cond_base**(i+1))), int(self.c_base*(cond_base**(i+1))), k_size_unet, self.scaling, nonlin="lrelu") for i in reversed(range(self.n_downscale))])
        self.up_b = torch.nn.ModuleList([MyConv(int(self.c_base*(cond_base**(i+1))), int(self.c_base*(cond_base**i)), k_size_unet, self.scaling, nonlin="lrelu") for i in reversed(range(self.n_downscale))])
        self.up_c = torch.nn.ModuleList([MyConv(int(self.c_base*(cond_base**(i+1)))+int(self.c_base*(cond_base**i)), int(self.c_base*(cond_base**i)), k_size_unet, self.scaling, nonlin="linear") for i in reversed(range(self.n_downscale))])

        self.bottom_c = self.down_b[-1].w.shape[1]
        self.bottom = MyConv(self.bottom_c, self.bottom_c, 1 if self.res_lowest == 1 else 3, self.scaling, nonlin="lrelu")



        self.conv_scale_out = MyConv(self.c_base//2, self.c_base//2, 3, self.scaling, nonlin="lrelu", init_zeros=False)
        self.conv_scale_out_b = MyConv(self.c_base//2, c_out//2, 1, self.scaling, nonlin="linear", init_zeros=True)

        self.conv_bias_out = MyConv(self.c_base//2, self.c_base//2, 3, self.scaling, nonlin="lrelu", init_zeros=False)
        self.conv_bias_out_b = MyConv(self.c_base//2, c_out//2, 1, self.scaling, nonlin="linear", init_zeros=True)


        self.resolutions = [self.res]
        self.channels = [self.c_base]
        temp = self.res
        j = 0
        for _ in self.down:
            for k in range(self.n_downscale_per_layer):
                temp = temp//2
            j = j + 1
            if j < len(self.down):
                self.channels.append(self.down[j].w.shape[1])
            else:
                self.channels.append(self.down[j-1].w.shape[0])
            self.resolutions.append(temp)



    def instance_norm(self, x):
        x_f = x.reshape(x.shape[0], x.shape[1], -1)
        n = torch.sqrt(torch.var(x_f, dim=-1 if x.shape[-1]>1 else -2)+1e-6).reshape(x.shape[0], x.shape[1] if x.shape[-1]>1 else 1, 1, 1)
        n_b = torch.mean(x_f, dim=-1 if x.shape[-1]>1 else -2).reshape(x.shape[0], x.shape[1] if x.shape[-1]>1 else 1, 1, 1)
        x = (x-n_b)/n
        return x
 

    def upsample(self, x):
        return self.us(x) if self.ds is not None and x.shape[-1] > 4 else torch.nn.functional.interpolate(x, size=2*x.shape[-1], mode="bilinear", align_corners=True)

    def downsample(self, x):
        return self.ds(x) if self.ds is not None and x.shape[-1] > 4 else torch.nn.functional.interpolate(x, size=x.shape[-1]//2, mode="bilinear", align_corners=True)

    def forward(self, x, cond=None, global_cond=None):

        if cond is not None and self.isCond:
            cond = self.conv_cond_connect_pre(cond)

            x = x*self.mask_in
            if self.masking_type == "vhalfs":
                x = torch.flip(x, (-1,))
            if self.masking_type == "hhalfs":
                x = torch.flip(x, (-2,))

            x = torch.cat([x, cond], axis=1)
            x = self.conv_cond_connect_un(x)

        else:
            x = x*self.mask_in


        x = self.conv_1(x)
        x = self.conv_2(x)

        x_arr = []
        for conv_a, conv_b, conv_c in zip(self.down, self.down_b, self.down_c):
            x_arr.append(x)
            for _ in range(self.n_downscale_per_layer):
                x = self.downsample(x)
            if x.shape[1] >= 2:
                x = self.instance_norm(x)
            h = x
            h = conv_a(h)
            h = conv_b(h)
            x = torch.cat([x,h], dim=1)
            x = conv_c(x)


        x = self.bottom(x)


        for conv_a, conv_b, conv_c in zip(self.up, self.up_b, self.up_c):

            for _ in range(self.n_downscale_per_layer):
                x = self.upsample(x)
            if x.shape[1] >= 2:
                x = self.instance_norm(x)
            h = x
            h = conv_a(h)
            h = conv_b(h)
            x = torch.cat([x,h], dim=1)
            x = conv_c(x)
            xx = x_arr.pop()
            x = (x + xx)/np.sqrt(2)

        x = self.conv_3(x)

        s,b = torch.chunk(x, 2, dim=1)

        s = self.conv_scale_out(s)
        s = self.conv_scale_out_b(s)*self.mask_out
        s = s.clip(-3.0,3.0)

        b = self.conv_bias_out(b)
        b = self.conv_bias_out_b(b)*self.mask_out
        b = b.clip(-5.0,5.0)

        return s, b


class CouplingCNN(torch.nn.Module):

    def __init__(self, c_in, c_out, c_middle, res,
                 isCond=False, ind=0, preproc_out_c=3, kernel_size=3, masking_type="channels", init_unit=False):
        super(CouplingCNN, self).__init__()

        k_size = kernel_size
        if res < 2:
            k_size = 1
        self.scaling = 1.0

        self.ind = ind

        self.c_mid = c_middle

        self.c_in = c_in
        self.c_out = c_out
        self.res = res
        self.isCond = isCond
        self.masking_type = masking_type


        self.masking_type = masking_type



        self.mask_in = torch.ones(size=(1, self.c_in, self.res, self.res))


        if masking_type == "channels":
            out_channels = int(np.ceil((self.c_in*0.5)))
            in_channels = self.c_in - out_channels
            self.mask_in[:,in_channels:,...] = 0.0
        elif masking_type == "spatial":
            self.mask_in[:,:,0::2, 0::2] = 0.0
            self.mask_in[:,:,1::2, 1::2] = 0.0
        elif masking_type == "blocks":
            m_ones = torch.ones(size=(1, self.c_in, 2, 2))
            m_zeros = torch.zeros(size=(1, self.c_in, 2, 2))
            m = torch.cat([torch.cat([m_ones, m_zeros], dim=-1), torch.cat([m_zeros, m_ones], dim=-1)], dim=-2)
            R = self.res//m.shape[-1]
            if R < 1:
                raise ValueError("Checkerboard-pattern needs at least resolution 4, is {}".format(self.res))
            self.mask_in = m.repeat(1,1,R,R)
        else:
            raise NotImplementedError("Masking type: {}".format(self.masking_type))
        self.mask_out = torch.where(self.mask_in>0.0, torch.zeros_like(self.mask_in), torch.ones_like(self.mask_in)) #see only the channels that were not seen at input

        if self.res == 1:
            self.mask_in = self.mask_in[:,:,0,0]
            self.mask_out = self.mask_out[:,:,0,0]
        self.mask_in = torch.nn.Parameter(self.mask_in.float(), requires_grad=False)
        self.mask_out = torch.nn.Parameter(self.mask_out.float(), requires_grad=False)

        if masking_type != "channels" and (ind%2)==0:
            self.mask_in, self.mask_out = self.mask_out, self.mask_in

        self.mask_in.data = self.mask_in.data*np.sqrt(2.0)



        if self.res > 1:
            self.flow_input_id = torch.nn.Identity()
            self.flow_input_id._myind = self.ind

        if self.res > 1:
            if not isCond:
                self.conv_0 = MyConv(self.c_in, self.c_in, k_size, self.scaling, nonlin="tanh")

            self.conv_1 = MyConv(self.c_in, self.c_mid, k_size, self.scaling, nonlin="lrelu")
            self.conv_2 = MyConv(self.c_mid, self.c_mid, k_size, self.scaling, nonlin="lrelu")
            self.conv_3 = MyConv(self.c_mid, self.c_mid, k_size, self.scaling, nonlin="lrelu")

    
            self.conv_out_bias = MyConv(self.c_mid//2, self.c_mid//2, 3, self.scaling, nonlin="lrelu")
            self.conv_out_bias_b = MyConv(self.c_mid//2, self.c_out//2, 1, self.scaling, nonlin="linear", init_zeros=init_unit)
            self.conv_out_scale = MyConv(self.c_mid//2, self.c_out//2, 1, self.scaling, nonlin="linear", init_zeros=True)

            if isCond:
                self.cond_preproc_a = MyConv(preproc_out_c, preproc_out_c, 3, self.scaling, nonlin="lrelu")
                self.conv_cond_connect = MyConv(c_in+preproc_out_c, c_in, 3, self.scaling, nonlin="tanh")
        else:
            if not isCond:
                self.conv_0 = MyLinear(self.c_in, self.c_in, self.scaling, nonlin="tanh")
            self.conv_1 = MyLinear(self.c_in, self.c_mid, self.scaling, nonlin="lrelu") 
            self.conv_2 = MyLinear(self.c_mid, self.c_mid, self.scaling, nonlin="lrelu")
            self.conv_3 = MyLinear(self.c_mid, self.c_mid, self.scaling, nonlin="lrelu")
    
            self.conv_out_bias = MyLinear(self.c_mid//2, self.c_mid//2, self.scaling, nonlin="lrelu") 
            self.conv_out_bias_b = MyLinear(self.c_mid//2, self.c_out//2, self.scaling, nonlin="linear", init_zeros=init_unit) 
            self.conv_out_scale = MyLinear(self.c_mid//2, self.c_out//2, self.scaling, nonlin="linear", init_zeros=True)

            if isCond:
                self.cond_preproc_a = MyLinear(preproc_out_c, preproc_out_c, self.scaling, nonlin="lrelu")
                self.conv_cond_connect = MyLinear(self.c_in+preproc_out_c, c_in, self.scaling, nonlin="lrelu")


    def instance_norm(self, x):
        x_f = x.reshape(x.shape[0], x.shape[1], -1)
        n = torch.sqrt(torch.var(x_f, dim=-1 if x.shape[-1]>1 else -2)+1e-6).reshape(x.shape[0], x.shape[1] if x.shape[-1]>1 else 1, 1, 1)
        n_b = torch.mean(x_f, dim=-1 if x.shape[-1]>1 else -2).reshape(x.shape[0], x.shape[1] if x.shape[-1]>1 else 1, 1, 1)
        x = (x-n_b)/n
        return x
 


    def forward(self, x, cond=None, global_cond=None):
        if self.res == 1:
            x = x.reshape(x.shape[0], -1)
        if cond is not None and self.isCond:
            if self.res == 1:
                cond = cond.reshape(cond.shape[0], -1)
            x = self.mask_in * x

            cond = cond + self.cond_preproc_a(cond)
            x = torch.cat([x, cond], axis=1)
            x = self.conv_cond_connect(x)
        else:
            x = self.mask_in*x
            x = self.conv_0(x)

        if self.res > 1:
            x = self.flow_input_id(x)

        if self.res > 1:
            x = self.instance_norm(x)

        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)

        s,b = torch.chunk(x, 2, dim=1)
        s = self.conv_out_scale(s)*self.mask_out
        s = s.clip(-3,3)

        b = self.conv_out_bias(b)
        b = self.conv_out_bias_b(b)*self.mask_out
        b = b.clip(-5,5)


        if self.res == 1:
            s = s[...,None,None]
            b = b[...,None,None]
        return s,b



class UnitaryPermute(torch.nn.Module):


    def __init__(self, channels, M=1.0):
        super(UnitaryPermute, self).__init__()
        self.c = channels
        self.M = M

        self._U = torch.nn.Parameter(torch.randn(self.c, self.c)/self.M)

        self.K_inv = None

    @property
    def U(self):
        return torch.matrix_exp((self._U-self._U.T)*self.M)

    @property
    def K(self):
        return torch.matrix_exp((self._U-self._U.T)*self.M)



    def compute_K_inv(self):
        self.K_inv = self.K.T



    def forward(self, x):

        K = self.U.T.reshape(self.c, self.c, 1, 1)
        x = torch.nn.functional.conv2d(x, K)
        return x, 0.0



    def inverse(self, x):
        K = self.U.reshape(self.c, self.c,1,1)
        x = torch.nn.functional.conv2d(x, K)
        return x, 0.0


    def ildj(self):
        return 0.0


class Permute(torch.nn.Module):

    def __init__(self, channels, M=1.0, init_identity=False):
        super(Permute, self).__init__()
        self.c = channels
        self.M = M


        init_mat = np.linalg.qr(np.random.normal(size=(self.c, self.c)))[0].astype(np.float32)/self.M

        if init_identity:
            init_mat = np.eye(self.c).astype(np.float32)

        self.K = torch.nn.Parameter(torch.from_numpy(init_mat), requires_grad=True)
        self.K_inv = None # Once in eval mode, we can just cache this and not do inversion every time


    def compute_K_inv(self):
        self.K_inv = torch.inverse(self.K.double()).float()

    def forward(self, x):

        if self.K_inv is not None:
            K = self.K_inv
        else:
            K = torch.inverse(self.K.double()).float()


        if x.shape[-1] == 1:
            x = torch.matmul(x[:,:,0,0], torch.t(K))/self.M
            x = x[...,None,None]
        else:
            K = K.reshape(self.c, self.c, 1, 1)
            x = torch.nn.functional.conv2d(x, K)/self.M
        detout = 0.0 # ldj is computed in batch elsewhere

        return x, detout

    def ildj(self):
        ildj = torch.slogdet(self.M*self.K.reshape(1, self.c, self.c))[1]
        return ildj.reshape(1,1)

    def inverse(self, x):

        K = self.K*self.M
        if x.shape[-1] == 1:
            x = torch.matmul(x[:,:,0,0], torch.t(K))
            x = x[...,None,None]
        else:
            x = torch.nn.functional.conv2d(x, K.reshape(self.c, self.c, 1, 1))
        detout = 0.0 # ldj is computed in batch elsewhere
        return x, detout



class Actnorm(torch.nn.Module):


    def __init__(self, c_in, res):
        super(Actnorm, self).__init__()
        self.res = res
        self.c = c_in

        self.an_scale = torch.nn.Parameter(torch.zeros([1, c_in, 1, 1]), requires_grad=True)
        self.an_bias = torch.nn.Parameter(torch.zeros(1, c_in, 1, 1, dtype=torch.float32), requires_grad=True)
        self.initialized = False


    def initialize(self, x):
        if torch.sum(torch.abs(self.an_bias)) > 1e-4:
            # Hack for handling EMA initialization.
            # The  "this.initialized" is False since the EMA network is not run before the first time visualizations are run.
            # The bias should not be exactly zero at that point so we just set the flag here.
            self.initialized = True
            return

        if self.res == 1:
            # zero-init when resolution == 0
            self.initialized = True
            return
        with torch.no_grad():
            # Data-based init using the current batch.
            x = x.permute(0,2,3,1)
            x = x.reshape(-1, x.shape[-1])
            mu = torch.mean(x, dim=0)
            std = torch.std(x, dim=0)
            self.an_bias.data = mu.reshape(1,-1,1,1)
            self.an_scale.data = torch.log(torch.abs(std.reshape(1,-1,1,1)))
        self.initialized = True


    def forward(self, x):
        x = x * torch.exp(self.an_scale) + self.an_bias

        fldj = self.an_scale.reshape(-1)
        fldj = torch.sum(fldj).reshape(1,1)

        fldj = fldj * self.res * self.res

        return x, fldj

    def inverse(self, x):
        if not self.initialized and self.an_bias.requires_grad:
            self.initialize(x)

        x = (x-self.an_bias)*torch.exp(-self.an_scale)

        ildj = self.an_scale.reshape(-1)
        ildj = -torch.sum(ildj).reshape(1,1)
        ildj = ildj * self.res * self.res

        return x, ildj



class Step(torch.nn.Module):


    def __init__(self, num_channels, res, rnvp_mid_layers,
                 isCond=False, ind=0, use_pooling_layers="", 
                 preproc_out_c=3, permute_kernel_scaling=1.0, rnvp_kernel_size=3,
                 unet_n_downscales=2, parent_ds_us=None, unet_c_mult=2.0, flow_split_type="channels",
                 use_unitary_permute=False):
        super(Step, self).__init__()

        self.res = res
        self.c = num_channels
        self.preproc_out_c = preproc_out_c
        self.use_unitary_permute = use_unitary_permute
        self.masking_type = flow_split_type
        self.ind = ind
        self.coupling_pred_logs = True


        if self.c == 1 and flow_split_type == "channels":
            raise ValueError("Cannot split in channel-direction when num-channels = 1")



        self.actnorm = Actnorm(num_channels, res)


        if (not use_pooling_layers) or self.res < 4:
            self.coupling = CouplingCNN(self.c, self.c*2, rnvp_mid_layers, res, isCond=isCond, ind=self.ind,
                                        preproc_out_c=self.preproc_out_c, kernel_size=rnvp_kernel_size,
                                        masking_type=flow_split_type)
        else:
            self.coupling = PoolingCouplingCNN(self.c, self.c*2, rnvp_mid_layers, res, isCond=isCond, pooling_type=use_pooling_layers,
                                                ind=self.ind, preproc_out_c=self.preproc_out_c,
                                                kernel_size=rnvp_kernel_size, unet_n_downscales=unet_n_downscales,
                                                parent_ds_us=parent_ds_us, unet_c_mult=unet_c_mult, masking_type=flow_split_type)





        if self.use_unitary_permute:
            self.invconv = UnitaryPermute(self.c, M=permute_kernel_scaling)
        else:
            self.invconv = Permute(self.c, M=permute_kernel_scaling)
        self.invconv_res = self.res
        self.invconv_c = self.c



    def inverse(self, x, cond=None, global_cond=None):

        x, ildj_tot = self.actnorm.inverse(x)

        s,b = self.coupling(x, cond, global_cond)
        x = (x-b)*torch.exp(-s) # masking at coupling to make this work
        ildj = -torch.sum(s, dim=(1,2,3)).view(-1, 1)
        ildj_tot = ildj_tot +  ildj

        x, _ = self.invconv.inverse(x) #the ildj is added elsewhere!


        return x, ildj_tot

    def forward(self, x, cond=None, global_cond=None):

        x, _ = self.invconv.forward(x) # the fldj is added elsewhere!

        s,b = self.coupling(x, cond, global_cond)
        x = torch.exp(s)*x + b
        fldj_tot = torch.sum(s, dim=(1,2,3)).view(-1, 1)


        x, fldj = self.actnorm.forward(x)
        fldj_tot = fldj_tot + fldj

        return x, fldj_tot



class ResolutionSteps(torch.nn.Module):

    def __init__(self, in_channels, res, L, rnvp_mid_layers,
                 isCond=False,
                 use_pooling_layers="",
                 preproc_out_c=3, permute_kernel_scaling=1.0, rnvp_kernel_size=3,
                 unet_n_downscales=2, parent_ds_us=None, unet_c_mult=2.0, flow_split_type="channels",
                 use_unitary_permute=False):
        super(ResolutionSteps, self).__init__()
        self.c = in_channels
        self.res = res
        self.preproc_out_c = preproc_out_c
        self.skip_forward_logdet = False
        self.identity = torch.nn.Identity()
        self.use_unitary_permute = use_unitary_permute
        self.permutation_kernel_scaling = permute_kernel_scaling


        if flow_split_type == "blocks-channels":
            flow_split_type = ["blocks" for _ in range(L//2)] + ["channels" for _ in range(L//2)] 
        elif flow_split_type == "hblocks-lblocks-blocks-channels":
            flow_split_type = ["hblocks" for _ in range(L//4)] + ["lblocks" for _ in range(L//4)] + ["blocks" for _ in range(L//4)] + ["channels" for _ in range(L//4)]
        elif "-" not in flow_split_type:
            flow_split_type = [flow_split_type for _ in range(L)]
        else:
            raise ValueError("Unknown split-type: {}".format(flow_split_type))

        if "blocks_only" in use_pooling_layers:
            self.use_pooling_layers = list(map(lambda x: "unet_only" if ("blocks" in x or "halfs" in x or "lblocks" in x or "hblocks" in x) else "", flow_split_type))
        else:
            self.use_pooling_layers = [use_pooling_layers for _ in range(L)]

        if "reduced" in use_pooling_layers:
            self.use_pooling_layers = list(map(lambda x: x+"_reduced" if x else "", self.use_pooling_layers))
        if "double" in use_pooling_layers:
            self.use_pooling_layers = list(map(lambda x: x+"_double" if x else "", self.use_pooling_layers))


        self.steps = torch.nn.ModuleList([Step(in_channels, res, rnvp_mid_layers,
                                               isCond=isCond,
                                               ind=i,
                                               use_pooling_layers=self.use_pooling_layers[i],
                                               preproc_out_c=self.preproc_out_c,
                                               permute_kernel_scaling=permute_kernel_scaling,
                                               rnvp_kernel_size=rnvp_kernel_size,
                                               unet_n_downscales=unet_n_downscales,
                                               parent_ds_us=parent_ds_us,
                                               unet_c_mult=unet_c_mult,
                                               flow_split_type=flow_split_type[i],
                                               use_unitary_permute=self.use_unitary_permute) for i in range(L)])



    def _permutationLogDet(self):
        # Compute matrix logdets in parallel with torch.slogdet instead of individually in each inverse-call.

        perm_steps = list(filter(lambda x: x.invconv is not None, self.steps))
        if not perm_steps or self.use_unitary_permute:
            return 0.0

        res_correction = self.res*self.res
        m = self.permutation_kernel_scaling
        perm_mats = list(map(lambda x: x.invconv.K.reshape(1, self.c, self.c), perm_steps))
        perm_mats = torch.cat(perm_mats, dim=0)
        M_correction = self.c * np.log(m)

        t = torch.slogdet(perm_mats)[1]
        t = t + M_correction

        logdets = torch.sum(t)*res_correction
        return logdets.reshape(1,1)

    def forward(self, x, cond=None, cond_flow=None, global_cond=None):

        fldj_tot = 0.0


        if cond is not None:
            start_b, start_s, end_b, end_s = cond#torch.chunk(cond, 4, dim=1)

        if cond is not None:
            x = x * torch.exp(end_s)
            fldj_tot = fldj_tot + torch.sum(end_s.reshape(end_s.shape[0], -1), dim=1).reshape(-1,1)
            x = x + end_b



        for block in self.steps:
            x, fldj = block.forward(x, cond_flow, global_cond)
            fldj_tot = fldj_tot + fldj

        if cond is not None:
            x = x * torch.exp(start_s)
            fldj_tot = fldj_tot + torch.sum(start_s.reshape(start_s.shape[0], -1), dim=1).reshape(-1,1)
            x = x + start_b


        if not self.skip_forward_logdet:
            perm_logdet = self._permutationLogDet() 
            fldj_tot = fldj_tot - perm_logdet

        x = self.identity(x)
        return x, fldj_tot


    def inverse(self, x, cond=None, cond_flow=None, global_cond=None):
        ildj_tot = 0.0
        x = self.identity(x)

        if cond is not None:
            start_b, start_s, end_b, end_s = cond#torch.chunk(cond, 4, dim=1)

        if cond is not None:

            x = x - start_b
            x = x * torch.exp(-start_s)
            ildj_tot = ildj_tot - torch.sum(start_s.reshape(start_s.shape[0], -1), dim=1).reshape(-1,1)


        for block in reversed(self.steps):
            x, ildj = block.inverse(x, cond_flow, global_cond)
            ildj_tot = ildj_tot + ildj

        perm_logdet = self._permutationLogDet()
        ildj_tot = ildj_tot + perm_logdet


        if cond is not None:
            x = x - end_b
            x = x * torch.exp(-end_s)
            ildj_tot = ildj_tot - torch.sum(end_s.reshape(end_s.shape[0], -1), dim=1).reshape(-1,1)


        return x, ildj_tot


class MyConv(torch.nn.Module):

    def __init__(self, c_in, c_out, kernel, eff_lr, nonlin="lrelu", init_zeros=False, no_bias=False):
        super(MyConv, self).__init__()

        self.lrelu_slope = 0.1
        self.nonlin = nonlin

        if nonlin=="lrelu":
            self.gain = np.sqrt(2.0/(1+self.lrelu_slope**2))
        elif nonlin=="tanh":
            self.gain = 5.0/3.0
        else:
            self.gain = 1.0

        self.he = self.gain/np.sqrt(c_in*kernel*kernel)
        if init_zeros:
            self.w = torch.nn.Parameter(torch.zeros([c_out, c_in, kernel, kernel]))
        else:
            self.w = torch.nn.Parameter(torch.randn([c_out, c_in, kernel, kernel])/eff_lr)

        self.b = torch.nn.Parameter(torch.zeros([c_out]))

        self.w_eff_lr = self.he*eff_lr
        self.b_eff_lr = eff_lr

        if no_bias:
            self.b_eff_lr = 0.0
        self.k = kernel
        self.pad = (kernel-1)//2
        self.eff_lr = eff_lr



    def forward(self, x):
        w = self.w*self.w_eff_lr
        b = self.b*self.b_eff_lr

        x = torch.nn.functional.conv2d(x, w, bias=b, padding=self.pad)


        if self.nonlin == "linear":
            return x
        elif self.nonlin == "tanh":
            return torch.tanh(x)
        elif self.nonlin == "relu":
            return torch.nn.functional.relu(x)
        else:
            return torch.nn.functional.leaky_relu(x, negative_slope=self.lrelu_slope)




class MyLinear(torch.nn.Module):

    def __init__(self, c_in, c_out, eff_lr, bias_init=0.0, nonlin="lrelu", init_zeros=False, use_bias=True):
        super(MyLinear, self).__init__()

        self.lrelu_slope = 0.1
        self.gain = np.sqrt(2.0/(1+self.lrelu_slope**2))
        self.nonlin = nonlin
        if nonlin == "linear":
            self.gain = 1.0

        self.he = self.gain/np.sqrt(c_in)
        if init_zeros:
            self.w = torch.nn.Parameter(torch.zeros([c_out, c_in])/eff_lr)
        else:
            self.w = torch.nn.Parameter(torch.randn([c_out, c_in])/eff_lr)
        self.use_bias = use_bias
        
        if self.use_bias:
            self.b = torch.nn.Parameter(torch.full([c_out], np.float32(bias_init))/eff_lr)
        self.w_eff_lr = self.he*eff_lr
        self.b_eff_lr = eff_lr

    def forward(self, x):
        w = self.w*self.w_eff_lr
        b = self.b*self.b_eff_lr if self.use_bias else None

        x = torch.nn.functional.linear(x,w,bias=b)
        if self.nonlin == "linear":
            return x
        elif self.nonlin == "tanh":
            return torch.tanh(x)
        else:
            return torch.nn.functional.leaky_relu(x, negative_slope=self.lrelu_slope)



class GlobalStateNetwork(torch.nn.Module):

    def __init__(self, c_in, c_mid, c_out, eff_lr=1.0):
        super(GlobalStateNetwork, self).__init__()
        self.c_in = c_in
        self.c_mid = c_mid
        self.c_out = c_out
        self.lrelu_slope = 0.1
        self.eff_lr = eff_lr

        self.beta = 0.05
        self.phi = 1.0

        self.fc1 = MyLinear(self.c_in, self.c_mid, eff_lr)
        self.fc_rest = torch.nn.ModuleList([MyLinear(self.c_mid, self.c_mid, eff_lr) for _ in range(5)])

    def forward(self, x):
        x = x.reshape(x.shape[0],-1)

        x = self.fc1(x)
        for item in self.fc_rest:
            x = item(x)
        return x




class Unet(torch.nn.Module):

    def __init__(self, in_channels, out_channels, in_res=None, base_in=1.5, parent_ds_us=None, effective_lr=1.0, in_channels_proj=None, use_instnorm=True):
        super(Unet, self).__init__()
        self.max_resdown = max(min(4, 2 if in_res is None else int(np.log2(in_res))-2), 1)
        self.base = base_in
        base = self.base
        self.lrelu_slope = 0.1
        self.us = None if parent_ds_us is None else parent_ds_us[1]
        self.ds = None if parent_ds_us is None else parent_ds_us[0]
        k_size = 3
        self.in_res = in_res
        self.use_instnorm = use_instnorm

        if in_channels_proj is None:
            in_channels_proj = in_channels

        self.start = MyConv(in_channels, in_channels_proj, k_size, effective_lr)

        self.down = torch.nn.ModuleList([MyConv(int(in_channels_proj*(base**i)), int(in_channels_proj*(base**(i+1))), k_size, effective_lr) for i in range(self.max_resdown)])
        self.down_b = torch.nn.ModuleList([MyConv(int(in_channels_proj*(base**(i+1))), int(in_channels_proj*(base**(i+1))), k_size, effective_lr) for i in range(self.max_resdown)])


        self.bottom = MyConv(int(in_channels_proj*(base**self.max_resdown)), int(in_channels_proj*(base**self.max_resdown)), 1, effective_lr)


        self.up = torch.nn.ModuleList([MyConv(int(in_channels_proj*(base**(i+1))), int(in_channels_proj*(base**(i))), k_size, effective_lr) for i in reversed(range(self.max_resdown))])
        self.up_b = torch.nn.ModuleList([MyConv(int(in_channels_proj*(base**i)), int(in_channels_proj*(base**(i))), k_size, effective_lr) for i in reversed(range(self.max_resdown))])


        self.end_a = MyConv(in_channels_proj, in_channels_proj, k_size, effective_lr, nonlin="lrelu", init_zeros=False)
        self.end_b = MyConv(in_channels_proj, out_channels, k_size, effective_lr, nonlin="linear", init_zeros=False)


        self.resolutions = [self.in_res]
        self.channels = [in_channels_proj]
        temp = self.in_res
        j = 0
        for _ in self.down:

            j = j + 1
            if j < len(self.down):
                self.channels.append(self.down[j].w.shape[0])
            else:
                self.channels.append(self.down[j-1].w.shape[0])

            temp = temp//2
            self.resolutions.append(temp)




    def instance_norm(self, x):
        x_f = x.reshape(x.shape[0], x.shape[1], -1)
        n = torch.sqrt(torch.var(x_f, dim=-1 if x.shape[-1]>1 else -2)+1e-6).reshape(x.shape[0], x.shape[1] if x.shape[-1]>1 else 1, 1, 1)
        n_b = torch.mean(x_f, dim=-1 if x.shape[-1]>1 else -2).reshape(x.shape[0], x.shape[1] if x.shape[-1]>1 else 1, 1, 1)
        x = (x-n_b)/n
        return x



    def forward(self, x):

        x = self.start(x)
        skips = [x]

        for layer_a, layer_b in zip(self.down, self.down_b):
            if self.use_instnorm:
                x = self.instance_norm(x)
            x = layer_a(x)
            x = layer_b(x)
            if self.ds is not None and x.shape[-1] > 8:
                x = self.ds(x)
            else:
                x = torch.nn.functional.interpolate(x, size=x.shape[-1]//2, mode="bilinear", align_corners=False)
            skips.append(x)
        _ = skips.pop()

        x = self.bottom(x)


        for layer_a, layer_b in zip(self.up, self.up_b):
            if self.us is not None and x.shape[-1] > 8:
                x = self.us(x)
            else:
                x = torch.nn.functional.interpolate(x, size=x.shape[-1]*2, mode="bilinear", align_corners=False)

            if self.use_instnorm:
                x = self.instance_norm(x)
            x = layer_a(x)
            x = layer_b(x)

            y = skips.pop()
            x = (x+y)/np.sqrt(2.0)

        x = self.end_a(x)
        return self.end_b(x)


class EncoderDecoder(torch.nn.Module):


    def __init__(self, in_res, out_res, c_in_encoder, c_out_decoder, z_c, c_hidden_encoder, c_hidden_decoder=None, k=3, gs_c=None, decoder_rel_noise=0.1, parent_ds_func=None, parent_us_func=None, skip_noise=False,
                block_y0=True, encoder_lr_scale=1.0, aug_type="noise", decoder_lr=1.0, c_skip=8,
                is_endblock=False, get_sibling_f=None, parent_ind=0):
        super(EncoderDecoder, self).__init__()

        n = int(np.log2(in_res//out_res))
        self.decoder_lr = decoder_lr
        self.aug_type = aug_type
        self.us = parent_us_func
        self.get_sibling_f = get_sibling_f


        self.decoder_input_id = torch.nn.Identity()
        self.decoder_cond_id = torch.nn.Identity()

        if n == 2:
            self.components = torch.nn.ModuleList([EncoderDecoderBlock(in_res, c_in_encoder, c_out_decoder, c_hidden_encoder, c_hidden_encoder, c_hidden_decoder=c_hidden_decoder, k=k, gs_c=gs_c, decoder_rel_noise=decoder_rel_noise, parent_ds_func=parent_ds_func, parent_us_func=parent_us_func,
                                                              skip_noise=True, block_y0=block_y0, encoder_lr_scale=encoder_lr_scale, aug_type=aug_type,
                                                              decoder_lr=decoder_lr, ind_from_lowres=1, ind_max=n-1, c_skip=c_skip, compute_flow_residuals=True, is_endblock=is_endblock,
                                                              parent_ind=parent_ind),
                                                   EncoderDecoderBlock(in_res//2, c_hidden_encoder, c_hidden_encoder, z_c, c_hidden_encoder, c_hidden_decoder=c_hidden_decoder, k=k, gs_c=gs_c, decoder_rel_noise=decoder_rel_noise,
                                                              skip_noise=skip_noise, parent_ds_func=parent_ds_func, parent_us_func=parent_us_func,
                                                              block_y0=block_y0, encoder_lr_scale=encoder_lr_scale, aug_type=aug_type,
                                                              decoder_lr=decoder_lr, ind_from_lowres=0, ind_max=n-1, c_skip=c_skip, is_endblock=is_endblock,
                                                              parent_ind=parent_ind)])
        elif n == 1:
            self.components = torch.nn.ModuleList([EncoderDecoderBlock(in_res, c_in_encoder, c_out_decoder, z_c, c_hidden_encoder, c_hidden_decoder=c_hidden_decoder, k=k, gs_c=gs_c, decoder_rel_noise=decoder_rel_noise, skip_noise=skip_noise,
                                                              parent_ds_func=parent_ds_func, parent_us_func=parent_us_func, block_y0=block_y0,
                                                              encoder_lr_scale=encoder_lr_scale, aug_type=aug_type,
                                                              decoder_lr=decoder_lr, ind_max=n-1, c_skip=c_skip, compute_flow_residuals=True, is_endblock=is_endblock,
                                                              parent_ind=parent_ind)])
        else:
            raise ValueError("There should be at most x4 change in resolution and at least x2 change.")

        for item in self.components:
            item.get_sibling_f = self.get_sibling_f



    def encode(self, x):
        for T in self.components:
            x,t = T.encode(x)
        return x,t

    def decode(self, x, gs=None, cond_prev=None):
        x = self.decoder_input_id(x)
        cond_prev = self.decoder_cond_id(cond_prev)

        for T in reversed(self.components):
            x, x_side = T.decode(x, gs, cond_prev)
        return x, x_side



class EncoderDecoderBlock(torch.nn.Module):

    def __init__(self, in_res, c_in_encoder, c_out_decoder, c_latent, c_hidden_encoder, c_hidden_decoder, k=3, gs_c=None, decoder_rel_noise=0.1, skip_noise=False, parent_ds_func=None, parent_us_func=None, block_y0=True,
                 encoder_lr_scale=1.0, aug_type="NOISE", decoder_lr=1.0, ind_from_lowres=0, ind_max=0, c_skip=8, compute_flow_residuals=False, is_endblock=False, parent_ind=0):
        super(EncoderDecoderBlock, self).__init__()

        self.res_in_encoder = in_res
        self.res_in_decoder = self.res_in_encoder//2
        self.c_in_encoder = c_in_encoder
        self.c_out_decoder = c_out_decoder
        self.c_latent = c_latent
        self.c_skip = c_skip
        self.lrelu_slope = 0.1

        self.encoder_noise_active = True


        
        self.is_lowres_in_block = ind_from_lowres == 0
        self.is_highres_in_block = ind_from_lowres == ind_max
        self.compute_flow_residuals = compute_flow_residuals


        self.ind_from_lowres = ind_from_lowres


        self.is_endblock = is_endblock
        self.ignore_mid_t = ind_max > 0 and ind_from_lowres > 0
        self.parent_ind = parent_ind


        self.c_hidden_encoder = c_hidden_encoder
        self.c_hidden_decoder = c_hidden_decoder


        self.aug_type = aug_type
        self.decoder_rel_noise = decoder_rel_noise
        self.skip_noise = skip_noise
        self.block_y0 = block_y0


        self.ds = parent_ds_func
        self.us = parent_us_func



        self.k = k
        if self.res_in_encoder < 4:
            self.k = 1
        self.p = (self.k-1)//2
        self.gs_c = gs_c

        self.enc_lr_scale = encoder_lr_scale
        self.decoder_lr_scale = decoder_lr



        cc_in = self.c_latent+self.c_skip if "LAST" not in self.aug_type else self.c_latent 

        if (self.gs_c is not None and self.gs_c > 0) and self.res_in_decoder > 2:
            if (self.res_in_encoder >= 4 or not self.block_y0) and not "NO_STYLE" in self.aug_type:
                self.A_1_s = MyLinear(self.gs_c, 2*self.c_hidden_decoder, 1.0, bias_init=0.0, nonlin="linear")
            if (self.res_in_encoder > 4 or not self.block_y0) and not "NO_STYLE" in self.aug_type:
                self.A_2_s = MyLinear(self.gs_c, 2*self.c_hidden_decoder, 1.0, bias_init=0.0, nonlin="linear")
        else:
            self.aug_type = self.aug_type + "_NO_STYLE"
 
        if self.res_in_decoder == 2: #i.e. this is the compressor closest to y0
            self.conv_d_const = torch.nn.Parameter(torch.randn(1, self.c_hidden_decoder, 4, 4, dtype=torch.float32), requires_grad=True)


        use_inorm = "NO_NORM" not in self.aug_type



        #### Encoder
        ####-------------
        self.conv_e1 = MyConv(self.c_in_encoder, self.c_hidden_encoder, self.k, self.enc_lr_scale)
        self.conv_e2 = MyConv(self.c_hidden_encoder, self.c_hidden_encoder, self.k, self.enc_lr_scale)

        if not self.is_endblock or (self.is_endblock and ind_from_lowres > 0):
            self.conv_e3 = MyConv(self.c_hidden_encoder, self.c_latent, 1, self.enc_lr_scale, nonlin="linear")
        if not self.is_endblock and not self.ignore_mid_t or self.is_endblock and self.ind_from_lowres==0:
            self.conv_e3_b = MyConv(self.c_hidden_encoder, self.c_hidden_encoder, 1, self.enc_lr_scale, nonlin="linear")
        ####-------------



        #### Decoder
        ####-------------


        encoder_only = self.res_in_decoder < 4
        if not encoder_only:
            k_d = 3

            cc_in = self.c_skip+self.c_latent if "LAST" not in self.aug_type else self.c_latent 

            if "LAST" not in self.aug_type and self.is_lowres_in_block:
                self.conv_d_pred_0 = Unet(self.c_skip+self.c_latent, self.c_latent, self.res_in_decoder, base_in=2.0, parent_ds_us=(self.ds, self.us) if self.ds is not None else None, effective_lr=1.0, use_instnorm=use_inorm)
                self.conv_d_pred_1 = MyConv(self.c_latent, self.c_hidden_decoder, 3, self.decoder_lr_scale, nonlin="lrelu", no_bias=False)
                self.conv_d_pred_2 = MyConv(self.c_hidden_decoder, self.c_hidden_decoder, 3, self.decoder_lr_scale, nonlin="lrelu", no_bias=True)
                self.conv_d_pred_3 = MyConv(self.c_hidden_decoder, self.c_latent, 3, self.decoder_lr_scale, nonlin="linear", init_zeros=False, no_bias=True)


            self.conv_d1 = MyConv(cc_in if self.is_lowres_in_block else self.c_hidden_decoder, self.c_hidden_decoder, k_d, self.decoder_lr_scale)
            self.conv_d2 = MyConv(self.c_hidden_decoder, self.c_hidden_decoder, k_d, self.decoder_lr_scale)
            self.conv_d3 = MyConv(self.c_hidden_decoder, self.c_hidden_decoder, k_d, self.decoder_lr_scale)
            self.conv_d4 = MyConv(self.c_hidden_decoder, self.c_hidden_decoder, k_d, self.decoder_lr_scale)



            if self.compute_flow_residuals:

                self.conv_d5_a = MyConv(self.c_hidden_decoder, self.c_out_decoder, 1, self.decoder_lr_scale, nonlin="linear")

                if "NOEUNET" not in self.aug_type and self.res_in_encoder > 8:
                    self.conv_d5_un = Unet(self.c_out_decoder, self.c_out_decoder, self.res_in_encoder, parent_ds_us=(self.ds, self.us) if self.ds is not None else None, effective_lr=1.0, use_instnorm=use_inorm)
                self.conv_d5_b = MyConv(self.c_hidden_decoder, self.c_out_decoder, 3, self.decoder_lr_scale, init_zeros=False)
                self.conv_d5_bb = MyConv(self.c_out_decoder, self.c_out_decoder, 1, self.decoder_lr_scale, nonlin="linear", init_zeros=True)


                self.conv_d5_c = MyConv(self.c_hidden_decoder, self.c_out_decoder, 1, self.decoder_lr_scale, nonlin="linear")
                if "NOEUNET" not in self.aug_type and self.res_in_encoder > 8:
                    self.conv_d5_c_un = Unet(self.c_out_decoder, self.c_out_decoder, self.res_in_encoder, parent_ds_us=(self.ds, self.us) if self.ds is not None else None, effective_lr=1.0, use_instnorm=use_inorm)
                self.conv_d5_d = MyConv(self.c_hidden_decoder, self.c_out_decoder, 3, self.decoder_lr_scale, init_zeros=False)
                self.conv_d5_dd = MyConv(self.c_out_decoder, self.c_out_decoder, 1, self.decoder_lr_scale, nonlin="linear", init_zeros=True)





    def noise_f(self, z):

        std = torch.mean(torch.sqrt(torch.var(z.reshape(z.shape[0], -1), dim=0)))

        if self.training and not self.skip_noise and "NOISE" in self.aug_type:
            noise = torch.randn_like(z)*std*self.decoder_rel_noise
            z = z + noise

        return z


    def instance_norm(self, x):
        x_f = x.reshape(x.shape[0], x.shape[1], -1)
        n = torch.sqrt(torch.var(x_f, dim=-1 if x.shape[-1]>1 else -2)+1e-6).reshape(x.shape[0], x.shape[1] if x.shape[-1]>1 else 1, 1, 1)
        n_b = torch.mean(x_f, dim=-1 if x.shape[-1]>1 else -2).reshape(x.shape[0], x.shape[1] if x.shape[-1]>1 else 1, 1, 1)
        x = (x-n_b)/n
        return x



    def encode(self, x):

        x = self.conv_e1(x)

        if self.ds is not None and x.shape[-1] > 4:
            x = self.ds(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)


        if x.shape[-1] > 1:
            x = self.instance_norm(x)

        x = self.conv_e2(x)

        if x.shape[-1] > 1:
            x = self.instance_norm(x)


        x_mid = self.conv_e3_b(x) if not self.is_endblock and not self.ignore_mid_t or (self.is_endblock and self.ind_from_lowres==0) else None
        x = self.conv_e3(x) if not self.is_endblock or (self.is_endblock and self.ind_from_lowres > 0) else None

        if x is not None and x.shape[-1] > 1:
            x = self.instance_norm(x)

        return x, x_mid# (to_flow, continue-to-lower_res) 

    def decode(self, z, gs=None, cond_last=None):

        if z.shape[-1] < 2:
            return z.repeat(1,1,2,2), None

        # just const with style mode at 4x4 -outputting layer (input is 2x2 hence the check res < 4)
        if z.shape[-1] < 4 and self.block_y0:
            x = self.conv_d_const.expand(z.shape[0], -1, -1, -1)
            if gs is not None and "NO_STYLE" not in self.aug_type:
                x_f = x.reshape(x.shape[0], x.shape[1], -1)
                n = torch.sqrt(torch.var(x_f, dim=-1)+1e-6).reshape(x.shape[0], x.shape[1], 1, 1)
                n_b = torch.mean(x_f, dim=-1).reshape(x.shape[0], x.shape[1], 1, 1)
                x = (x-n_b)/n
                s = self.A_1_s(gs)
                s,b = torch.chunk(s, 2, dim=1)
                x = x*(s.reshape(x.shape[0], -1, 1, 1)) + b.reshape(x.shape[0], -1, 1, 1)
            return x,None
        elif z.shape[-1] < 4 and not self.block_y0:
            x = z.repeat(1,1,2,2) + self.conv_d_const.expand(z.shape[0], -1, -1, -1)
            return x,None
        else:
            pass


        z = self.noise_f(z) # only adds noise when .training is true


        if cond_last is not None and self.ind_from_lowres == 0:


            z = (z + self.conv_d_pred_0(torch.cat([cond_last, z], dim=1)))/np.sqrt(2.0)

            w = self.conv_d_pred_1(z)
            w = (w + self.conv_d_pred_2(w))/np.sqrt(2.0)
            w = self.conv_d_pred_3(w)

            z = torch.cat([cond_last, w], dim=1)


        x = self.conv_d1(z)

        if not "NO_NORM" in self.aug_type:
            x = self.instance_norm(x)

        x = (self.conv_d2(x) + x)/np.sqrt(2.0)

        if self.us is not None:
            x = self.us(x)
        else:
            x = torch.nn.functional.interpolate(x, size=x.shape[-1]*2, mode="nearest")



        if gs is not None and "NO_STYLE" not in self.aug_type:
            x = self.instance_norm(x)
            s = self.A_1_s(gs)
            s,b = torch.chunk(s, 2, dim=1)
            x = x*(s.reshape(x.shape[0], -1, 1, 1)) + b.reshape(x.shape[0], -1, 1, 1)
        else:
            if "NO_NORM" not in self.aug_type:
                x = self.instance_norm(x)
            else:
                pass



        x = (self.conv_d3(x) + x)/np.sqrt(2.0)



        if gs is not None and "NO_STYLE" not in self.aug_type:
            x = self.instance_norm(x)
            s = self.A_2_s(gs)
            s,b = torch.chunk(s, 2, dim=1)
            x = x*(s.reshape(x.shape[0], -1, 1, 1)) + b.reshape(x.shape[0], -1, 1, 1)
        else:
            if "NO_NORM" not in self.aug_type:
                x = self.instance_norm(x)
            else:
                pass
   

        x = (self.conv_d4(x) + x)/np.sqrt(2.0)

        if not self.compute_flow_residuals and "NO_NORM" not in self.aug_type:
            x = self.instance_norm(x)


        if self.compute_flow_residuals:

            xx =  x
            start_b = self.conv_d5_a(xx)

            if "NOEUNET" not in self.aug_type and self.res_in_encoder >8:
                start_b = (self.conv_d5_un(start_b) + start_b)/np.sqrt(2.0)

            start_s = self.conv_d5_b(xx)
            start_s = self.conv_d5_bb(start_s)
            start_s = start_s.clip(-3,3)

            end_b = self.conv_d5_c(xx)

            if "NOEUNET" not in self.aug_type and self.res_in_encoder > 8:
                end_b = (self.conv_d5_c_un(end_b) + end_b)/np.sqrt(2.0)

            end_s = self.conv_d5_d(xx)
            end_s = self.conv_d5_dd(end_s)
            end_s = end_s.clip(-3,3)

            x_residuals = (start_b, start_s, end_b, end_s)
        else:
            x_residuals = None




        if not self.compute_flow_residuals:
            return x, None
        else:
            return x_residuals, x


class PartitionedEndblock(torch.nn.Module):


    def __init__(self, parent_prior, parent_noise_f, channels_total=0, channels=[], layers=[], rnvp_channels=[], cond_output_channels=[], permute_scaling=1.0, noise_scales=0.1, noise_scale_rel=1.0,
                mask_for_style_ind=[], mask_for_base_ind=[], use_unitary_permute=True):
        super(PartitionedEndblock, self).__init__()

        channels[-1] = int(channels_total - np.sum(np.array(channels[0:-1])))
        self.c = channels

        self.endblock_uncond = ResolutionSteps(channels[0], 1, layers[0], rnvp_channels[0], isCond=False, use_pooling_layers="", preproc_out_c=None, permute_kernel_scaling=permute_scaling, rnvp_kernel_size=1, use_unitary_permute=use_unitary_permute)
        conds_c = list(accumulate(channels[:-1]))
        
        temp = list(map(lambda x: ResolutionSteps(x[0], 1, x[1], x[2], isCond=True, use_pooling_layers="", preproc_out_c=x[3], permute_kernel_scaling=permute_scaling, rnvp_kernel_size=1, use_unitary_permute=use_unitary_permute), zip(channels[1:], layers[1:], rnvp_channels[1:], cond_output_channels[1:])))
        self.conditionals = torch.nn.ModuleList(temp) if temp else []

        self.cond_encoders = torch.nn.ModuleList(list(map(lambda x: MyConv(x[0], x[1], 1, 1.0, nonlin="linear"), zip(conds_c, cond_output_channels[1:]))))

        self.prior = parent_prior
        self.noise_std_f = parent_noise_f
        self.noise_scales = noise_scales
        self.noise_scale_rel = noise_scale_rel
    
        self.loss_w = 1.0

        mask_for_style = np.ones(channels_total)
        mask_for_style[mask_for_style_ind[0]:mask_for_style_ind[1]] = 0.0
        self._mask_for_style = torch.nn.Parameter(torch.from_numpy(mask_for_style.astype(np.float32)).reshape(1,-1), requires_grad=False)


        mask_for_base = np.ones(channels_total)
        mask_for_base[mask_for_base_ind[0]:mask_for_base_ind[1]] = 0.0
        self._mask_for_base = torch.nn.Parameter(torch.from_numpy(mask_for_base.astype(np.float32)).reshape(1,-1), requires_grad=False)

        self.encoder_noise_active = True


    def mask_style(self, x):
        if len(x.shape) > 2:
            return x*self._mask_for_style[:,:,None,None]
        else:
            return x*self._mask_for_style
    def mask_base(self, x):
        if len(x.shape) > 2:
            return x*self._mask_for_base[:,:,None,None]
        else:
            return x*self._mask_for_base

    def toInternal(self, x):
        if not self.conditionals:
            return [x, None]
        return torch.split(x, self.c, dim=1)

    def fromInternal(self, x):
        if not self.conditionals:
            return x[0]
        return torch.cat(x, dim=1)


    def entropy(self, x, std):
        var = std*std
        k = np.prod(x.shape[1:])

        perdim = 0.5*torch.log(2.0*np.pi*var) + 0.5
        return k*perdim


    def inverse(self, y):
        y = self.toInternal(y)
        z0, ildj_tot = self.endblock_uncond.inverse(y[0])
        out = [z0]

        for yy,flow,encoder, this_ind in zip(y[1:], self.conditionals, self.cond_encoders, range(len(self.conditionals))):
            cond = torch.cat(y[:this_ind+1],dim=1)
            cond = encoder(cond)
            z,ildj = flow.inverse(yy, cond_flow=cond)
            out.append(z)
            ildj_tot = ildj_tot + ildj

        out = self.fromInternal(out)
        return out, ildj_tot

    def apply_noise(self, x):
        y = self.toInternal(x)
        y = list(map(lambda x: x[0]+self.noise_std_f(x[0])*torch.randn_like(x[0])*x[1]*self.noise_scale_rel, zip(y,self.noise_scales)))
        return self.fromInternal(y)

    def loss(self, y):

        y = self.toInternal(y)

        std = self.noise_std_f(y[0])*self.noise_scales[0]
        y0_noise = std*torch.randn_like(y[0])

        if self.encoder_noise_active:
            y0_in = y[0] + y0_noise
        ent = self.entropy(y0_in, std)
        ent = ent.detach()

        z0, ildj = self.endblock_uncond.inverse(y0_in.detach())


        p = self.prior(z0)
        uc = p + ildj + ent # log-prior + encoder-entropy, negated later to arrive at a KL-estimate
        prior_tot = 0.0
        ildj_tot = 0.0

        ll_tot = -(p+ildj+ent)/(np.prod(z0.shape[1:])*np.log(2.0))


        for yy,flow,encoder, this_ind,noise_scale in zip(y[1:], self.conditionals, self.cond_encoders, range(len(self.conditionals)), self.noise_scales[1:]):
            cond = torch.cat(y[:this_ind+1],dim=1)
            cond = encoder(cond)

            std = self.noise_std_f(yy)*noise_scale
            ent = self.entropy(yy, std)
            ent = ent.detach()


            if self.encoder_noise_active:
                y_noise = std*torch.randn_like(yy)
                y_in = yy + y_noise


            z,ildj = flow.inverse(y_in.detach(), cond_flow=cond)

            ildj = ildj + ent


            p = self.prior(z)
            ll = -(p+ildj)/(np.prod(z.shape[1:])*np.log(2.0))
            ll_tot = ll_tot + ll
            prior_tot = prior_tot + p

            ildj_tot = ildj + ildj_tot

        return ll_tot*self.loss_w, prior_tot, ildj_tot, uc


    def flow_forward(self, z):
        z = self.toInternal(z)
        y0, fldj_tot = self.endblock_uncond.forward(z[0])
        out = [y0]

        for yy,flow,encoder, this_ind in zip(z[1:], self.conditionals, self.cond_encoders, range(len(self.conditionals))):
            cond = torch.cat(out[:this_ind+1],dim=1)
            cond = encoder(cond)
            y,fldj = flow.forward(yy, cond_flow=cond)
            out.append(y)
            fldj_tot = fldj_tot + fldj

        out = self.fromInternal(out)
        return out, fldj_tot


class MultiStep(torch.nn.Module):


    def __init__(self, resolutions=[], channels=[], skips_channels=[], loss_weights=[], num_layers=[], encoder_channels=[], decoder_channels=[], encoder_decoder_kernels=[], rnvp_channels=[], rnvp_kernel_sizes=[], unet_c_mult=[], permute_scaling=[], pooling_type=[], unet_n_downscales=[],
            flow_split_types=[], cond_out_c=[], global_net_channels=256,  augmentation_types=[], decoder_rel_noise=[], encoder_lr_scale=1.0, decoder_lr_scale=1.0,
            endblock_settings={}, prior_stds=[], use_unitary_permute=[]):
        super(MultiStep, self).__init__()

        self.resolutions = resolutions + [min(4,resolutions[-1]//2)]
        self.endres = endblock_settings.endres
        self.endres_c = endblock_settings.endblock_tot_channels//(self.endres**2)

        self.channels = channels + [endblock_settings.autoencoder_channels]
        if self.resolutions[-2] != 8:
            raise ValueError("Lowest-resolution flow with res>1 should be 8x8.")



        self.weights = loss_weights
        self.init_scale = 4.0 # data variance is approximately 1 after multiplication with 4


        is_style_main = any(list(map(lambda t: "NO_STYLE" not in t, augmentation_types)))
        if not is_style_main:
            global_net_channels = -1
            print("==== disabled style network, since it would not be used")





        self.global_net_channels = global_net_channels > 0
        self.global_state_out = global_net_channels if self.global_net_channels else -1
        self.prior_stds = prior_stds
        self.device = "cuda:0" #placeholder


        self.decoder_rel_noise = [None] + decoder_rel_noise # Offset of one: the native-resolution flow does not add noise to the data and does not have a higher-resolution decoder
        self.skips_channels = [1] + skips_channels
        self.aug_type = [augmentation_types[0]] + augmentation_types



        k1d = bil()
        k = k1d.reshape(-1,1,1)*k1d.reshape(1,-1,1)
        k = k.transpose(2,0,1)[None,...]
        self._interp_K = torch.from_numpy(k.astype(np.float32))
        self.register_buffer("interp_K", self._interp_K)

        k1d = bil()
        k = k1d.reshape(-1,1,1)*k1d.reshape(1,-1,1)
        k = k.transpose(2,0,1)[None,...]
        self._interp_K_down = torch.from_numpy(k.astype(np.float32))
        self.register_buffer("interp_K_down", self._interp_K_down)




        self.endblock_autoencoder = EncoderDecoder(self.resolutions[-1], self.endres, encoder_channels[-1], endblock_settings.autoencoder_channels, self.endres_c, endblock_settings.autoencoder_channels, endblock_settings.autoencoder_channels,
                                                         k = 3, gs_c=self.global_state_out,
                                                         decoder_rel_noise=0.0, # this has no effect anyway since skip_noise=True below
                                                         parent_ds_func=None, parent_us_func=None, block_y0=endblock_settings.block_y0,
                                                         encoder_lr_scale=encoder_lr_scale,
                                                         aug_type=endblock_settings.augmentation_type + "_LAST",
                                                         decoder_lr=decoder_lr_scale,
                                                         skip_noise=True, # should not add noise within this encoder--decoder
                                                         c_skip=1,
                                                         is_endblock=True) 




        endblock_settings.channels[-1] = self.endres*self.endres*self.endres_c-np.sum(np.array(endblock_settings.channels[:-1]))
        self.endblock_unified = PartitionedEndblock(self.prior, self._style_noise, self.endres*self.endres*self.endres_c, endblock_settings.channels, endblock_settings.layers, endblock_settings.rnvp_channels,
                                                    endblock_settings.cond_output_channels, endblock_settings.permute_scaling, endblock_settings.style_rel_noises, endblock_settings.style_rel_noise_scale,
                                                    mask_for_base_ind=endblock_settings.mask_for_base_ind,
                                                    mask_for_style_ind=endblock_settings.mask_for_style_ind, use_unitary_permute=endblock_settings.use_unitary_permute)



        self.endblock_split_conv = MyConv(self.endres*self.endres*self.endres_c, self.endres*self.endres*self.endres_c, 1, 1.0, nonlin="linear")


        mask_inds = endblock_settings.mask_for_style_ind
        if (mask_inds[1]-mask_inds[0] >= self.endres*self.endres*self.endres_c or not self.global_net_channels):
            self.global_state = torch.nn.Identity()
        else:
            self.global_state = GlobalStateNetwork(self.endres*self.endres*self.endres_c, endblock_settings.global_state_hidden_channels, self.global_state_out, endblock_settings.mapping_network_lr_scale)


        self.blocks = torch.nn.ModuleList([OneStep(self.resolutions[ind], self.resolutions[ind+1], 3 if ind==0 else encoder_channels[ind-1],
                                                   self.channels[ind], self.channels[ind+1],
                                                   encoder_channels[ind],
                                                   decoder_channels[ind],
                                                   encoder_decoder_kernels[ind],
                                                   num_layers[ind],
                                                   kernel_size=rnvp_kernel_sizes[ind],
                                                   rnvp_channels=rnvp_channels[ind],
                                                   permute_scaling=permute_scaling[ind],
                                                   parent_gs_c_out=self.global_state_out,
                                                   pooling=pooling_type[ind],
                                                   decoder_rel_noise=self.decoder_rel_noise[ind+1],
                                                   flow_rel_noise=self.decoder_rel_noise[ind],
                                                   is_output=ind==0,
                                                   is_last=ind==len(self.resolutions)-2,
                                                   unet_n_downscales=unet_n_downscales[ind],
                                                   encoder_lr_scale=encoder_lr_scale,
                                                   cond_out_c=cond_out_c[ind],
                                                   decoder_aug_type=self.aug_type[ind+1],
                                                   flow_aug_type=self.aug_type[ind],
                                                   decoder_lr_scale=decoder_lr_scale,
                                                   unet_c_mult=unet_c_mult[ind],
                                                   flow_split_type=flow_split_types[ind],
                                                   ind=ind,
                                                   c_skip_in=self.skips_channels[ind+1],
                                                   c_skip_out=self.skips_channels[ind],
                                                   prior_std=self.prior_stds[ind],
                                                   use_unitary_permute=use_unitary_permute[ind],
                                                   get_sibling_f=self.get_sibling) for ind in range(len(self.resolutions)-1)])



    def set_device(self, rank):
        self.device = "cuda:{}".format(rank)

    @property
    def fullres(self):
        return self.resolutions[0]


    def non_encoder_parameters(self):
        out = []
        for k,p in self.named_parameters():
            if "conv_e" not in k and "conv_d" not in k:
                out.append(p)
        return out

    def decoder_parameters(self):
        out = []
        for k,p in self.named_parameters():
            if "conv_d" in k:
                out.append(p)
        return out

    def encoder_parameters(self):
        out = []
        for k,p in self.named_parameters():
            if "conv_e" in k:
                out.append(p)
        return out


    def inverse(self, x):
        out, ildj = self.layerwise_inverse(x)

        return out, reduce(lambda x,y: x+y, ildj)


    def layerwise_inverse(self, x):
        x = x*self.init_scale
        out = []
        out_ildj = []
        targets = self.create_scales(x)

        t = targets[-1].reshape(-1, self.endres_c*self.endres*self.endres, 1, 1)

        z0, ildj = self.endblock_unified.inverse(t)
        out_ildj.append(ildj)
        out.append(z0)


        gs = self.global_state(self.endblock_unified.mask_style(t)).reshape(x.shape[0], -1)
        cond = self.endblock_split_conv(self.endblock_unified.mask_base(t))

        cond_prev = None
        cond,_ = self.endblock_autoencoder.decode(cond, gs)

        for block, target in zip(reversed(self.blocks), reversed(targets[:-2])):
            z,ldj,cond_prev = block.inverse(target, cond=cond, gs=gs, cond_prev=cond_prev)
            out_ildj.append(ldj)
            out.append(z)
            cond = target

        return list(reversed(out)), list(reversed(out_ildj))




    def flow_forward(self, Z):

        z0 = Z[-1].reshape(Z[-1].shape[0], -1, 1, 1)
        y0,fldj = self.endblock_unified.flow_forward(z0)


        gs = self.global_state(self.endblock_unified.mask_style(y0)).reshape(y0.shape[0], -1)
        cond = self.endblock_split_conv(self.endblock_unified.mask_base(y0))


        cond,_ = self.endblock_autoencoder.decode(cond, gs)

        cond_prev = None
        for z, block in zip(reversed(Z[:-1]), reversed(self.blocks)):
            cond,ldj,cond_prev = block.forward(z, cond, gs=gs, cond_prev=cond_prev)
            fldj = fldj + ldj


        return cond[:,:3,:,:]/self.init_scale, fldj


    def forward(self, x):
        return self.loss(x)



    def create_scales(self, x, scale=False):
        if scale:
            x = x*self.init_scale

        targets = [x]
        t = x

        for block in self.blocks:
            out, t = block.learned_upscaler.encode(t)
            targets.append(out)

        _,out = self.endblock_autoencoder.encode(t)
        targets.append(out)

        return targets

    def _style_noise(self, t):
        std = torch.mean(torch.sqrt(torch.var(t.reshape(t.shape[0], -1), dim=0)))
        return std

    def loss(self, x, return_layerwise=None):

        x = x*self.init_scale


        targets = self.create_scales(x)


        t = targets[-1].reshape(x.shape[0], self.endres_c*self.endres*self.endres, 1, 1)

        if return_layerwise is not None:
            if not ("tot" in return_layerwise):
                return_layerwise["tot"] = {}
                return_layerwise["neg_logprior"] = {}
                return_layerwise["fldj"] = {}
                return_layerwise["normalization"] = 0
                return_layerwise["ll_unscaled"] = []



        ll_tot, prior_tot, ildj_tot, uc = self.endblock_unified.loss(t)

        ll_tot_unscaled = prior_tot + ildj_tot + uc
        normalization = np.prod(t.shape[1:])


        # logging
        if return_layerwise is not None:
            nn = (self.endres_c-self.endblock_unified.c[0])*self.endres*self.endres*np.log(2.0)
            nn_uc = self.endblock_unified.c[0]*self.endres*self.endres*np.log(2.0)
            if "1_endblock" in return_layerwise["tot"]:
                return_layerwise["tot"]["1_endblock"].append(-(prior_tot+ildj_tot).detach().cpu().numpy()/nn)
                return_layerwise["neg_logprior"]["1_endblock"].append(-prior_tot.detach().cpu().numpy()/nn)
                return_layerwise["fldj"]["1_endblock"].append(-ildj_tot.detach().cpu().numpy()/nn)
            else:
                return_layerwise["tot"]["1_endblock"] = [-(prior_tot+ildj_tot).detach().cpu().numpy()/nn]
                return_layerwise["neg_logprior"]["1_endblock"] = [-prior_tot.detach().cpu().numpy()/nn] 
                return_layerwise["fldj"]["1_endblock"] = [-ildj_tot.detach().cpu().numpy()/nn]

            if "1_endblock_uncond" in return_layerwise["tot"]:
                return_layerwise["tot"]["1_endblock_uncond"].append(-uc.detach().cpu().numpy()/nn_uc)
            else:
                return_layerwise["tot"]["1_endblock_uncond"] = [-uc.detach().cpu().numpy()/nn_uc]



        t = self.endblock_unified.apply_noise(t)

        gs = self.global_state(self.endblock_unified.mask_style(t)).reshape(t.shape[0], -1)
        cond = self.endblock_split_conv(self.endblock_unified.mask_base(t))

        cond,_ = self.endblock_autoencoder.decode(cond, gs)
        cond_prev = None

        for block, target, loss_w in zip(reversed(self.blocks), reversed(targets[:-2]), reversed(self.weights)):

            
            y, ldj, estimate = block.inverse(target, cond=cond, gs=gs, cond_prev=cond_prev)

            prior = block.prior(y, sigma=block.prior_std)
            cond_prev = estimate
            cond = target
         

            normalization = normalization + np.prod(target.shape[1:])


            if y.shape[-1] == self.resolutions[0]:
                ldj = ldj + np.log(self.init_scale)*self.resolutions[0]*self.resolutions[0]*3

            ll = -(prior + ldj)/(np.prod(target.shape[1:])*np.log(2.0)) * loss_w
            ll_tot_unscaled = ll_tot_unscaled + prior + ldj

            # logging
            if return_layerwise is not None:
                nn = np.prod(target.shape[1:])*np.log(2.0)
                if str(block.res) in return_layerwise["tot"]:
                    return_layerwise["tot"][str(block.res)].append(-(prior+ldj).detach().cpu().numpy()/nn)
                    return_layerwise["neg_logprior"][str(block.res)].append(-prior.detach().cpu().numpy()/nn)
                    return_layerwise["fldj"][str(block.res)].append(-ldj.detach().cpu().numpy()/nn)
                else:
                    return_layerwise["tot"][str(block.res)] = [-(prior+ldj).detach().cpu().numpy()/nn]
                    return_layerwise["neg_logprior"][str(block.res)] = [-prior.detach().cpu().numpy()/nn] 
                    return_layerwise["fldj"][str(block.res)] = [-ldj.detach().cpu().numpy()/nn]

            ll_tot = ll_tot + ll

        # logging
        if return_layerwise is not None:
            return_layerwise["normalization"] = (self.resolutions[0]**2)*3
            return_layerwise["ll_unscaled"].append(-ll_tot_unscaled.detach().cpu().numpy())

        if return_layerwise:
            return return_layerwise
        else:
            return ll_tot

    def sample(self, n, std=0.8):
        Z = [torch.randn(size=(n, self.channels[ind], r, r), dtype=torch.float32).to(self.device)*std*self.blocks[ind].prior_std for ind,r in enumerate(self.resolutions[:-1])] + [torch.randn(size=(n,self.endres_c,self.endres,self.endres), dtype=torch.float32).to(self.device)]

        X,_ = self.flow_forward(Z)
        x = X.detach().cpu().numpy()
        return x

    def _sample(self, n, std=1.0):
        Z = [torch.randn(size=(n, self.channels[ind], r, r), dtype=torch.float32).to(self.device)*std*self.blocks[ind].prior_std for ind,r in enumerate(self.resolutions[:-1])] + [torch.randn(size=(n,self.endres_c,self.endres,self.endres), dtype=torch.float32).to(self.device)]

        X,_ = self.flow_forward(Z)
        return X


    def sample_normalized(self, n, std=1.0):
        t = self._sample(n, std)
        t = (t + 0.5).clamp(0.0, 1.0)
        return t

    def prior(self, x, sigma=1.0):
        k = np.prod(x.shape[1:])

        xf = x.view(x.shape[0], -1)/sigma
        ll = -torch.sum(xf*xf, dim=1).view(-1, 1)*0.5 - 0.5*float(k)*np.log(2.0*np.pi)- k*np.log(sigma)
        ll = ll.view(-1, 1)
        return ll


    def interpolate_up(self, x):
        return interpolate2_torch(x, self.interp_K)

    def interpolate_down(self, x):
        return decimate2_torch(x, self.interp_K_down)


    def set_noise_active(self, active):
        for block in self.blocks:
            for item in block.learned_upscaler.components:
                item.encoder_noise_active = active
            block.encoder_noise_active = active
        self.endblock_unified.encoder_noise_active = active


    def test_invert(self):
        self.eval()
        x = torch.rand(1, 3, self.resolutions[0], self.resolutions[0]).to(torch.device(self.device))
        z, ildj = self.inverse(x)

        x_back, fldj = self.flow_forward(z)
        x_o = x.detach().cpu().numpy()
        x_b = x_back.detach().cpu().numpy()
        ildj = ildj.detach().cpu().numpy()
        fldj = fldj.detach().cpu().numpy()

        self.train()
        print("Diff was: ", np.mean(np.square(x_o-x_b)))
        print("logdetJ diff was: ", np.mean(np.square(ildj+fldj)))


    def get_sibling(self, ind):
        if ind < len(self.blocks):
            return self.blocks[ind]
        else:
            return None


class OneStep(torch.nn.Module):


    def __init__(self, res, res_next, c_prev, c, c_out, encoder_channels, decoder_channels, upscale_kernel, L, kernel_size, rnvp_channels, permute_scaling, parent_gs_c_out, pooling="",
                 unet_n_downscales=2, decoder_rel_noise=0.1, flow_rel_noise=0.1, is_output=False, is_last=False, encoder_lr_scale=1.0, cond_out_c=32,
                 decoder_aug_type="NOISE", flow_aug_type="NOISE", decoder_lr_scale=1.0, unet_c_mult=2.0, flow_split_type="channels", ind=0, c_skip_in=8, c_skip_out=8,
                 prior_std=1.0, use_unitary_permute=False, get_sibling_f=None):
        super(OneStep, self).__init__()
        self.res = res
        self.res_next = res_next
        self.c_prev = c_prev
        self.c = c 
        self.c_out = c_out
        self.c_skip_in = c_skip_in
        self.c_skip_out = c_skip_out
        self.L = L
        self.flow_rel_noise = flow_rel_noise
        self.is_output = is_output
        self.is_last = is_last
        self.is_top = ind==0
        self.flow_aug_type = flow_aug_type


        self.get_sibling_f = get_sibling_f
        self.encoder_noise_active = True



        self.decoder_aug_type = decoder_aug_type
        if self.is_last:
            self.decoder_aug_type = self.decoder_aug_type + "_LAST"

        self.prior_std = prior_std





        k1d = bil()
        k = k1d.reshape(-1,1,1)*k1d.reshape(1,-1,1)
        k = k.transpose(2,0,1)[None,...]
        self._interp_K = torch.from_numpy(k.astype(np.float32))
        self.register_buffer("interp_K", self._interp_K)

        k1d = bil()
        k = k1d.reshape(-1,1,1)*k1d.reshape(1,-1,1)
        k = k.transpose(2,0,1)[None,...]
        self._interp_K_down = torch.from_numpy(k.astype(np.float32))
        self.register_buffer("interp_K_down", self._interp_K_down)





        self.learned_upscaler_cond = MyConv(decoder_channels, cond_out_c, 1, 1.0, nonlin="linear")
        if not self.is_output:
            self.skip_to_next = MyConv(decoder_channels, self.c_skip_out, 1, decoder_lr_scale, nonlin="linear")


        self.learned_upscaler = EncoderDecoder(self.res, self.res_next, self.c if self.is_top  else self.c_prev, self.c, self.c_out, encoder_channels, c_hidden_decoder=decoder_channels, k = upscale_kernel,
                                                     gs_c=parent_gs_c_out, decoder_rel_noise=decoder_rel_noise,
                                                     parent_ds_func=self.interpolate_down if not self.is_last else None, parent_us_func=self.interpolate_up,
                                                     skip_noise=self.is_last,
                                                     encoder_lr_scale=encoder_lr_scale, aug_type=self.decoder_aug_type,
                                                     decoder_lr=decoder_lr_scale, c_skip=self.c_skip_in,
                                                     is_endblock=is_last,
                                                     get_sibling_f=get_sibling_f,
                                                     parent_ind=ind)

        self.blocks = ResolutionSteps(self.c,
                                      self.res,
                                      L,
                                      rnvp_channels,
                                      isCond=True,
                                      use_pooling_layers=pooling,
                                      preproc_out_c=cond_out_c,
                                      permute_kernel_scaling=permute_scaling,
                                      rnvp_kernel_size=kernel_size,
                                      unet_n_downscales=unet_n_downscales,
                                      parent_ds_us=(self.interpolate_down, self.interpolate_up),
                                      unet_c_mult=unet_c_mult,
                                      flow_split_type=flow_split_type,
                                      use_unitary_permute=use_unitary_permute)




    def _add_noise(self, x, noise_scale):

        std = torch.mean(torch.sqrt(torch.var(x.reshape(x.shape[0], -1), dim=0)))

        ent = self.entropy(x, std*noise_scale)
        ent = ent.detach()

        if self.encoder_noise_active:
            noise = torch.randn_like(x)*std*noise_scale
            x = x + noise

        return x,ent


    def entropy(self, x, std):
        var = std*std
        k = np.prod(x.shape[1:])

        perdim = 0.5*torch.log(2.0*np.pi*var) + 0.5
        return k*perdim


    def inverse(self, x, cond=None, gs=None, cond_prev=None):
        

        start_end, cond_flow_raw = self.learned_upscaler.decode(cond, gs, cond_prev=cond_prev)
        cond_flow = self.learned_upscaler_cond(cond_flow_raw)


        if not self.is_output and "NOISE" in self.flow_aug_type:
            x, ent = self._add_noise(x, self.flow_rel_noise)
        else:
            ent = 0.0


        x, ildj = self.blocks.inverse(x.detach(), cond=start_end, cond_flow=cond_flow, global_cond=gs) # note the stop-grad -detach for the flow input 'x'


        ildj = ildj + ent # prior added later and negated for a KL / reconstruction term


        skip = self.skip_to_next(cond_flow_raw) if not self.is_output else None

        return x, ildj, skip


    def flow_forward(self, y, cond=None, gs=None, cond_prev=None):

        start_end, cond_flow_raw  = self.learned_upscaler.decode(cond, gs, cond_prev=cond_prev)
        cond_flow = self.learned_upscaler_cond(cond_flow_raw)

        x, fldj = self.blocks.forward(y, cond=start_end, cond_flow=cond_flow, global_cond=gs)

        skip = self.skip_to_next(cond_flow_raw) if not self.is_output else None

        return x, fldj, skip



    def forward(self, y, cond=None, gs=None, cond_prev=None):
        return self.flow_forward(y, cond=cond, gs=gs, cond_prev=cond_prev)

       

    def prior(self, x, sigma=1.0):
        k = np.prod(x.shape[1:])

        xf = x.view(x.shape[0], -1)/sigma
        ll = -torch.sum(xf*xf, dim=1).view(-1, 1)*0.5 - 0.5*float(k)*np.log(2.0*np.pi) - k*np.log(sigma)
        ll = ll.view(-1, 1)
        return ll



    def interpolate_up(self, x):
        return self._interpolate_up(x)

    def interpolate_down(self, x):
        return self._interpolate_down(x)

    def _interpolate_up(self, x):
        return interpolate2_torch(x, self.interp_K)

    def _interpolate_down(self, x):
        return decimate2_torch(x, self.interp_K_down)

    def test_invert(self):
        self.eval()
        x = torch.rand(5, 3, self.res, self.res).to(torch.device(self.device))
        z, ildj, _ = self.inverse(x, x)

        x_back, fldj, _ = self.flow_forward(z, x)
        x_o = x.detach().cpu().numpy()
        x_b = x_back.detach().cpu().numpy()
        ildj = ildj.detach().cpu().numpy()
        fldj = fldj.detach().cpu().numpy()

        self.train()
        print("Diff was: ", np.mean(np.square(x_o-x_b)))
        print("logdetJ diff was: ", np.mean(np.square(ildj+fldj)))

