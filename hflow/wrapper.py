import numpy as np
import torch
from PIL import Image
import os
import pickle
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

from hflow.imagegrid import ImageGrid
from hflow.utils import Network
from hflow.dataloader import ImageDataset
from hflow.metrics import compute_features_generator, compute_features_real, compute_fid, try_load_inception_net



class Wrapper(object):

    def __init__(self, path_to_model, config=None):
        """Wrapper around a torch-network for a hierarchical flow-model. Methods for sampling and evaluation.
        args:
            path_to_model: model-binary to be loaded
            config: dict of elements "INCEPTION_NET_PATH", "DATAPATH", "DS_NAME" for loading the inception net and a hdf5 dataset for FID computation (optional)
        """

        print("Loading: ", path_to_model)
        self._net = Network.load(path_to_model, dump_code=False)
        self.net = self._net.torch_network


        with torch.no_grad():
            try:
                self._compute_inverses()
                print("Compute inverses: success!")
            except:
                print("Failed to compute inverses - doing it manually at every forward pass...")

            try:
                self.set_logdet_computation(False)
                print("Forward log-det computation disabled!")
            except:
                print("Failed to disable forward log-det computation. Might cause slightly slower inference.")

        if config is not None:
            self.inception_net_path = config["INCEPTION_NET_PATH"]
            path_to_data = config["DATAPATH"]
            dataset_name = config["DS_NAME"]

            self.ds_train = ImageDataset(path_to_data, dataset_name, discretisation_bits=8, resize_to=self.res)

            self.ds_train.h_flip = False #disable flipping of images
        else:
            self.ds_train = None
            self.inception_net_path = None


        self.eval()
        self.net.set_noise_active(False) # disable the addition of noise making the encoding/decoding-process deterministic


    @property
    def res(self):
        return self.net.resolutions[0]

    @property
    def settings(self):
        if hasattr(self._net, "settings"):
            return self._net.settings
        else:
            return {}


    @classmethod
    def from_path(cls, path):
        """Load model without specifying a dataset-config
        """
        model = cls(path, None)
        return model


    def load_img(self, filename):
        with Image.open(filename) as img:
            img = img.resize((self.res, self.res), Image.LANCZOS)
            r = np.asarray(img).transpose(2,0,1).astype(np.float32)/255.0
            r = r - 0.5
            r = r[None,...]
            r_b = torch.from_numpy(r[:,0:3,...]).cuda().contiguous()
            return r_b


    def set_logdet_computation(self, val):
        """Are the log-det-Jacobians of the constituent Normalizing Flows computed during evaluation?
        """
        for block in self.net.blocks:
            block.blocks.skip_forward_logdet = val

        self.net.endblock_unified.endblock_uncond.skip_forward_logdet = val
        for block in self.net.endblock_unified.conditionals:
            block.skip_forward_logdet = val


    def sample_from_train_dataset(self, n_samples):
        """Sample random samples from dataset
        """
        if self.ds_train is None:
            raise RuntimeError("No dataset was loaded with the model, cannot sample reals.")
        inds = np.random.randint(low=0, high=len(self.ds_train), size=n_samples)
        r = np.concatenate([self.ds_train[i][0][np.newaxis,...] for i in inds], axis=0)
        r = torch.from_numpy(r).cuda()
        return r


    def get_from_train_dataset(self, ind, n_samples):
        """Load images [ind, ind+1,...ind+n_samples-1] from dataset
        """
        if self.ds_train is None:
            raise RuntimeError("No dataset was loaded with the model, cannot sample reals.")
        inds = np.arange(ind, ind+n_samples)
        r = np.concatenate([self.ds_train[i][0][np.newaxis,...] for i in inds], axis=0)
        r = torch.from_numpy(r).cuda()
        return r

    def get_inds_from_train_dataset(self, inds):
        """ Load images of indices from dataset
        """

        if self.ds_train is None:
            raise RuntimeError("No dataset was loaded with the model, cannot sample reals.")

        r = np.concatenate([self.ds_train[i][0][np.newaxis,...] for i in inds], axis=0)
        r = torch.from_numpy(r).cuda()
        return r

    def get_flow(self, ind):
        """Access an individual flow-block from the hierarchical model."""

        if ind == len(self.net.blocks) or ind == -1:
            return self.net.endblock
        else:
            return self.net.blocks[ind]


    def forward(self, z, uncond_last=False):
        """Map a latent code into an image. Inverse of the below .inverse-function.
        args:
            z : list of latent codes whose length and shapes are determined by the network -- see the output of .inverse() to figure out the shapes.
            uncond_last: is the list reversed to have order low-resolution latent codes --> high-resolution latent-codes
        out:
            torch tensor of shape [batch, 3, resolution, resolution] (batch of images in [-0.5, 0.5])
        """


        ee = len(self.net.endblock_unified.c)
        end = z[-ee:]

        if uncond_last:
            end = list(reversed(end))

        conc = self.net.endblock_unified.fromInternal(end)

        t = z[:-ee]
        z = t + [conc]

        x,_ = self.net.flow_forward(z)
        return x



    def inverse(self, x, uncond_last=False):
        """Map an image into the latent space of the model.
        args:
            x: Tensor of shape [batch, 3, resolution, resolution]. Should be manually normalized to [-0.5, 0.5]
            uncond_last: is the list reversed to have order low-resolution latent codes --> high-resolution latent-codes
        out:
            list of latent codes whose shapes depend on the model architecture
        """
        z,_ = self.net.inverse(x)
        end = z[-1]
        end = self.net.endblock_unified.toInternal(end)

        if uncond_last:
            end = list(reversed(end))
        out = z[:-1] + list(end)

        return out


    def _slerp_or_lerp(self, x, y, t, slerp=False):
        if torch.mean(torch.sum(torch.abs(x-y).reshape(x.shape[0], -1), dim=1)) < 1e-3:
            return x
        tt = t
        tt_i = 1.0-tt
        if not slerp:
            return tt_i*x + tt*y
        b = x.shape[0]
        x_n = x/torch.sqrt(torch.sum((x*x).reshape(b,-1),dim=1)).reshape(b,1,1,1)
        y_n = y/torch.sqrt(torch.sum((y*y).reshape(b,-1),dim=1)).reshape(b,1,1,1)
        cos_t = torch.sum((x_n*y_n).reshape(b,-1), dim=1).reshape(b,1,1,1)
        theta = torch.acos(cos_t)
        out = torch.sin(tt_i*theta)/torch.sin(theta)*x + torch.sin(tt*theta)/torch.sin(theta)*y
        return out

    def slerp_z(self, z_a, z_b, t, slerp=False, interp_in_y0=False):
        """Interpolation of lists latent codes z_a and z_b.
        """
        out = []

        z_a = self._split_style_cat(z_a)
        z_b = self._split_style_cat(z_b)

        if interp_in_y0:
            z_a[-1] = self.z_to_y0(z_a[-1])
            z_b[-1] = self.z_to_y0(z_b[-1])
        for a,b in zip(z_a, z_b):
            out.append(self._slerp_or_lerp(a, b, t, slerp=slerp))

        if interp_in_y0:
            out[-1] = self.y0_to_z(out[-1])

        out = self._split_style_split(out)
        return out


    def sample(self, n_images, std):
        """Sample the model
        args:
            n_images: number of samples
            std: float, standard deviation used when sampling the priors of the model

        out:
            np.array of shape [n_images, 3, resolution, resolution] of dtype np.float32, in the range [-0.5, 0.5]
            needs to be normalized e.g. with the Wrapper.to_image -function

        """
        with torch.no_grad():
            return self.net.sample(n_images, std)#[0]

    def _sample(self, n_images, std):
        """
        Same as above but returns a torch.tensor instead of numpy array
        """
        with torch.no_grad():
            return self.net._sample(n_images, std)


    def sample_z_prior_only(self, z, std):
        """ Only drawn the z_prior component of a latent code from the prior.
        """
        z_out = list(map(lambda x: torch.clone(x), z))
        z_out[-1] = torch.randn_like(z[-1])*std
        return z_out


    def _split_style_split(self, z):
        z_out = z[:-1]
        zz = z[-1]
        zz = list(self.net.endblock_unified.toInternal(zz))
        return z_out + zz


    def _split_style_cat(self, z):
        ee = len(self.net.endblock_unified.c)
        z_out = z[:-ee]
        z_rest = z[-ee:]
        zzab = self.net.endblock_unified.fromInternal(z_rest)
        return z_out + [zzab]


    def y0_to_z(self, y0):
        """Convert [z_1, z_prior] into [y_1, y_prior]
        """
        z0,_ = self.net.endblock_unified.inverse(y0)
        return z0


    def z_to_y0(self, z):
        """Inverse of the above method
        """
        y0,_ = self.net.endblock_unified.flow_forward(z)
        return y0



    def load_inception_net(self, path):
        return try_load_inception_net(path)


    def compute_fid(self, n_samples, fid_sample_std, bs=64, precompute_feature_path=None, compute_with_noise=True, verbose=False, save_features=True):
        """Compute the model FID using the dataset that is specified when a Wrapper is created
        args:
            n_samples: number of real and generated samples used when computing the FID
            fid_sample_std: list of standard deviations using which latent codes are drawn from the prior
            bs: batch size for generation
            precompute_feature_path: if real-image inception features have already been computed, use those instead. Otherwise, computes them again.
            compute_with_noise: Does the dataloader add noise to the reals to match the images used to train the model
            verbose: print progress?
            save_features: save the real-image inception features to be used later?
        out:
            FID as a float
        """

        if self.inception_net_path is None:
            raise RuntimeError("No Inception net instance was loaded with the model -- cannot computed FID.")
        if self.ds_train is None:
            raise RuntimeError("No dataset was loaded with the model, cannot compute FID.")


        inception = self.load_inception_net(self.inception_net_path)
        if inception is None:
            return -1.0

        if not compute_with_noise:
            t = self.ds_train.add_noise
            self.ds_train.add_noise = False

        self.ds_train.h_flip = False
        if verbose:
            print("FID-computation: Disabled dataset mirror-aug for FID-computation")
            print("FID-computation: Dataset discretization is: {} bits".format(self.ds_train._disc))

        iterator = DataLoader(self.ds_train, bs, shuffle=True, num_workers=0)



        features_gen = compute_features_generator(inception, self.net, bs, n_samples, sample_std=fid_sample_std, verbose=verbose)
        features_real = compute_features_real(inception, iterator, tot_shown_max=n_samples, cache_path_override=precompute_feature_path, noisy=compute_with_noise, verbose=verbose, save_features=save_features)
        fid = compute_fid(features_real, features_gen, max_samples=n_samples)


        if not compute_with_noise:
            self.ds_train.add_noise = t

        self.ds_train.h_flip = True
        if verbose:
            print("FID-computation: Enabled dataset mirror-aug after FID-computation")

        return fid



    def compute_fid_from_file(self, n_samples, path):
        """Compute FID using generated samples that are saved to a .npy-file at the argument path
        """

        if self.inception_net_path is None:
            raise RuntimeError("No Inception net instance was loaded with the model -- cannot computed FID.")
        if self.ds_train is None:
            raise RuntimeError("No dataset was loaded with the model, cannot compute FID.")

        net_path = self.inception_net_path
        with open(net_path, "rb") as f:
            inception = pickle.load(f).cuda()

        bs = 64
        data = np.load(path)
        print("Loaded shape: {} from path {}".format(data.shape, path))


        res = data.shape[1]
        dataset_gen = ImageDataset(path=None, ds=None, discretisation_bits=8, const_rand_scale=0, noise_std=0.0, noise_freq=0.0, noise_std_max_val=0.0, resize_to=res, data=(data,), split_percent=1.0, is_trainset=True)
        iterator_gen = DataLoader(dataset_gen, bs, shuffle=False, num_workers=0)
        features_gen = compute_features_real(inception, iterator_gen, tot_shown_max=n_samples, cache_path_override=None, noisy=False, verbose=True, save_features=False)


        dataset_real = ImageDataset(path=None, ds=None, discretisation_bits=8, const_rand_scale=0, noise_std=0.005, noise_freq=0.0, noise_std_max_val=0.0, resize_to=res, data=(self.ds_train.dataset,), split_percent=1.0, is_trainset=True)
        iterator = DataLoader(dataset_real, bs, shuffle=False, num_workers=0)
        features_real = compute_features_real(inception, iterator, tot_shown_max=n_samples, cache_path_override=None, noisy=False, verbose=True, save_features=False)


        fid = compute_fid(features_real, features_gen, max_samples=n_samples)


        print("FID was: {}".format(fid))
        return fid



    def simple_interpolation(self, data_inds_a, data_inds_b, n_timesteps, in_y0=False, override_a_with_z=None, override_b_with_z=None):

        """Interpolate between dataset images data_inds_a and data_inds_b
        args:
            data_inds_a/b: indices of images in a dataset
            n_timesteps: number of frames in interpolation
            in_y0: interpolate in the y0 space instead of z0
            override_a/b_with_z: instead of the dataset, use given latents.
        out:
            ImageGrid with block shape [1, timesteps]
        """

        a = self.get_inds_from_train_dataset(data_inds_a) if data_inds_a is not None else None
        b = self.get_inds_from_train_dataset(data_inds_b) if data_inds_a is not None else None
        if a is None or b is None:
            if override_a_with_z is None or override_b_with_z is None:
                raise RuntimeError("Must specify overriding latent codes if dataset indices are not specified.")

        res = a.shape[-1] if a is not None else override_a_with_z[0].shape[-1]

        with torch.no_grad():

            if override_a_with_z is not None:
                z_a = override_a_with_z
                assert override_b_with_z is not None
                z_b = override_b_with_z
            else:
                z_a = self.inverse(a)
                z_b = self.inverse(b)


            out = []
            for t in np.linspace(0.0, 1.0, n_timesteps):
                z = self.slerp_z(z_b, z_a, t, interp_in_y0=in_y0)

                x = self.forward(z)
                x = self.to_grid_row(x, res=res).reshape(1,-1).group()
                out.append(x)



        out = ImageGrid.stackV(out)
        return out


    def generate_reals_interpolation_video(self, data_inds_a, data_inds_b, n_timesteps, fps, output_filename, interpolate_in_y0=False):

        out = self.simple_interpolation(data_inds_a, data_inds_b, n_timesteps, in_y0=interpolate_in_y0)
        out.toVideo((1,1), mirror=True, fps=fps, filename=output_filename)



    def _gen_latent_safe(self, n, std=1.0):
        """Generate a latent code appropriate for the loaded model using the given standard deviation for the Gaussian priors
        args:
            n: number of samples
            std: float or list of floats. In case of a single float, it is broadcasted to all parts of the latent
        """
        cc = self.net.channels
        rr = self.net.resolutions[:-1]


        if type(std) is list:
            pass
        else:
            std = [std for _ in range(len(rr)+1)]

        Z = [torch.randn(size=(n, cc[ind], r, r), dtype=torch.float32).to(self.net.device)*std[ind]*self.net.blocks[ind].prior_std for ind,r in enumerate(rr)] + [torch.randn(size=(n,self.net.endres_c,self.net.endres,self.net.endres), dtype=torch.float32).to(self.net.device)*std[-1]]
        return Z


    def gen_individual_seeds(self, seeds, std, uncond_last=False):
        """Generate latent codes using the given seeds and stds.
        args:
            seeds: list of random generator seeds
            std: prior standard deviation
            uncond_last: does the output have latent codes from z_prior to z_highres or reversed?
        out:
            list of torch tensors to be used in .forward() for image generation.
        """
        with torch.no_grad():
            t = None
            for seed in seeds:
                torch.manual_seed(seed)
                z = self._gen_latent_safe(1, std)

                if t is None:
                    t = z
                else:
                    t = list(map(lambda x: torch.cat([x[0], x[1]], dim=0), zip(t, z)))

        t_end = t[-1]
        t_head = t[:-1]
        z0 = list(self.net.endblock_unified.toInternal(t_end))
        if len(z0) == 2 and z0[-1] is None:
            t = t_head + [z[0]]
        else:
            if uncond_last:
                z0 = list(reversed(z0))
            t = t_head + z0

        return t


    def sample_individual_seeds(self, seeds, std, label=True):
        """Sample the first image generated after settings torch.manual_seed to the elements of the input seeds-list.
        args:
            seeds: list of random generator seeds
            std: prior standard deviation
            label: write the seed to the corner of the image?
        out:
            ImageGrid of block shape [1, len(seeds)]
        """
        with torch.no_grad():
            out = []
            t = None
            res = self.net.resolutions[0]

            for seed in seeds:
                torch.manual_seed(seed)
                z = self._gen_latent_safe(1, std)
                if t is None:
                    t = z
                else:
                    t = list(map(lambda x: torch.cat([x[0], x[1]], dim=0), zip(t, z)))

                if len(t[0]) > 31:
                    x,_ = self.net.flow_forward(t)
                    out.append(self.to_grid_row(x, res=res))
                    t = None


            if t:
                x,_ = self.net.flow_forward(t)
                out.append(self.to_grid_row(x, res=res).reshape(1,-1))


        out = ImageGrid.stackV(out).reshape(1,-1)

        if label:
            out = out.split(axis=1)
            out = list(map(lambda x: x[0].caption_frame(str(x[1]), font_size=32, color=(255,0,0)), zip(out, seeds)))
            out = ImageGrid.stackV(out).reshape(1,-1)

        return out


    def _latent_space_random_walk(self, n_images, n_timesteps, n_interpolation, std, slerp=False, interp_in_y0=False, fix_layers=[], reals=False, reals_seed=0, generated_seeds=None):

        N = n_images[0]*n_images[1]

        with torch.no_grad():
            n = n_timesteps - 1
            x_prev = self._sample(N, 0.8)
            z_prev = self.inverse(x_prev)

            if type(std) is not list:
                std = [std for _ in z_prev]

            print("Std-list: ", std)


            seed_ind = 0
            if generated_seeds is not None:
                print(generated_seeds[0])
                z_prev = self.gen_individual_seeds(generated_seeds[0], std)
                seed_ind = seed_ind + 1
            else:
                z_prev = list(map(lambda x: torch.randn_like(x[0])*x[1], zip(z_prev, std)))



            if reals:
                np.random.seed(reals_seed)
                x_prev = self.sample_from_train_dataset(N)
                z_prev = self.inverse(x_prev)
            x_prev = self.forward(z_prev)

            w = n_images[1]
            h = n_images[0]

            if type(std) is not list:
                std = [std for _ in range(len(x_prev)-1)] + [1.0]

            out = []
            seed_offset = 1


            for _ in range(n):
                if generated_seeds is not None:
                    seeds = generated_seeds[seed_ind]
                    print(seeds)
                    seed_ind = seed_ind + 1
                    z_next = self.gen_individual_seeds(seeds, std)
                else:
                    z_next = list(map(lambda x: torch.randn_like(x[0])*x[1] if x[0].shape[-1] not in fix_layers else x[0], zip(z_prev, std)))


                x_next = self.forward(z_next)
                if reals:
                    np.random.seed(reals_seed+seed_offset)
                    seed_offset = seed_offset + 1
                    x_next = self.sample_from_train_dataset(n_images)
                    z_next = self.inverse(x_next)
                out.append(self.to_grid_row(x_prev, res=256).reshape(h,w).group())

                for t in np.linspace(0.0, 1.0, n_interpolation+1)[1:]:
                    x = self.forward(self.slerp_z(z_prev, z_next, t, slerp=slerp, interp_in_y0=interp_in_y0))
                    x = self.to_grid_row(x, res=256).reshape(h,w).group()
                    out.append(x)

                z_prev = z_next
                x_prev = x_next

        out = ImageGrid.stackV(out)

        return out


    def generate_latent_random_walk_video(self, n_images, n_endpoints, n_timesteps_per_interpolation, std, output_filename, fps):

        """Generate video of a random walk in the latent space.
        args:
            n_images: tuple (h,w) grid of independent walk paths shown in the output
            n_endpoints: number of vertices (= random points) in the latent space between which we interpolate
            n_timesteps_per_interpolation: number of frames generated between vertices
            std: standard deviation for the random gaussian latent samples
            output_filename: output file
            fps: frames-per-second in the output video

        out:
            None, as a side effect, saves a video of grid of images of shape (h, w)
        """

        out = self._latent_space_random_walk(n_images, n_endpoints, n_timesteps_per_interpolation, std)

        out.toVideo((1,1), fps=fps, filename=output_filename)



    def layerwise_latent_randomize(self, stds_orig=[], seed=0, n_samples=32, real=False):

        """Figure & Video for latent-layerwise, stddev / change on image, when the latent-code is randomized -- while fixing the other parts of the latent code.

        """

        with torch.no_grad():
            K = n_samples


            torch.manual_seed(seed)
            if real:
                x = self.get_inds_from_train_dataset([i for i in range(seed, seed+1)])
                z_orig = self.inverse(x, uncond_last=True)
            else:
                z_orig = self.gen_individual_seeds([seed], stds_orig, uncond_last=True)
                z_orig = list(map(lambda t: t.repeat(K, 1, 1, 1), z_orig))


            res = []
            frames = []


            for l in reversed(range(len(z_orig))):
                z = list(map(lambda t: torch.clone(t), z_orig))
                z[l] = torch.randn_like(z[l])
                xx = self.forward(z, uncond_last=True)

                t = torch.mean(torch.sqrt(torch.var(xx, axis=0)), axis=0) # variance over batch of images, mean over channels
                t = t.detach().cpu().numpy()

                res.append(t)
                frames.append(self.to_grid_row(xx).reshape(1,-1))



            res = np.concatenate(res, axis=0)
            res = res / np.max(res) # scale is normalized over changes of all the latent layers -- not individually, and not across models.

            frames = ImageGrid.stackH(frames)

            labels = list(map(lambda x: str(tuple(x.shape[1:])), reversed(z_orig)))


            frames.toVideo((1,len(z_orig)), fps=1, filename="latent_randomization.mp4")
            img = ImageGrid.from_bw(256, res, colormap="viridis").reshape(1,-1)
            img = img.caption_flat(labels, font_size=24)


            img.save("latent_randomization_.png")



    def latent_zeroing(self, real_inds=[]):

        """ Take images from a dataset, encode them into the latent and cumulatively set the latent codes to zero starting from high-resolution latent.

        """

        with torch.no_grad():
            x = self.get_inds_from_train_dataset(real_inds)

            z = self.inverse(x, uncond_last=True)

            x = self.to_grid_row(x)

            append_at = [256,128,32,8,1,1]

            for i in range(len(z)):
                z[i] = torch.zeros_like(z[i])
                if z[i].shape[-1] in append_at:
                    t = self.forward(z, uncond_last=True)
                    x = x * self.to_grid_row(t)

            x = x.transpose()
            labels = ["real", "+256", "+128", "+32", "+8", "+1uc", "mean-img"]
            x_labs = x.caption_flat(labels)
            x_labs.save("latentzero.png")


    def _compute_inverses(self):
        """Precompute 1x1 convolution kernel values for evaluation, as some of the are not directly optimized but parametrized to enforce orthonormality
        """

        if self.net.endblock_unified:
            eb = self.net.endblock_unified
            for step in eb.endblock_uncond.steps:
                if step.invconv is not None:
                    step.invconv.compute_K_inv()
            for cond in eb.conditionals:
                for step in cond.steps:
                    if step.invconv is not None:
                        step.invconv.compute_K_inv()
        else:
            eb = self.net.endblock.steps
            for step in eb:
                if step.invconv is not None:
                    step.invconv.compute_K_inv()

        for layer in self.net.blocks:
            for step in layer.blocks.steps:
                if step.invconv is not None:
                    step.invconv.compute_K_inv()

    def sample_with_std_list(self, std_list, n_samples, return_noise=False):
        """ Same as the .sample -function but allowing different standard deviations for different latent layers of the model.

        """
        with torch.no_grad():
            base = self.net._sample(n_samples, std=1.0)
            base = self.inverse(base)
            Z = list(map(lambda x: torch.randn_like(x[0])*x[1], zip(base, std_list)))


            if return_noise:
                return self.forward(Z), Z
            else:
                return self.forward(Z)


    def _normalize(self, x_np):
        x = (np.minimum(np.maximum(x_np+0.5, 0.0), 1.0)*255.0).astype(np.uint8)
        return x


    @staticmethod
    def normalize(x_np):
        x = (np.minimum(np.maximum(x_np+0.5, 0.0), 1.0)*255.0).astype(np.uint8)
        return x


    @staticmethod
    def to_image(x, scale=1):
        """ Convert a model output tensor/array into an ImageGrid-object. Automatically handles normalization.

        """
        res = x.shape[-1]

        if type(x) is not np.ndarray:
            x = x.detach().cpu().numpy()
        x = np.transpose(Wrapper.normalize(x), (0,2,3,1)).reshape(-1, res, 3)

        img = ImageGrid(x, res_h=res, res_w=res).reshape(1,-1)

        img = img.resize_by_scale(scale, method=Image.NEAREST)

        return img

    def to_grid_row(self, x, res=None):
        """ Same as above but not a static method.
        """
        if res is None:
            res = self.res
        x = x.detach().cpu().numpy()
        x = np.transpose(self._normalize(x), (0,2,3,1)).reshape(-1, self.res, 3)

        img = ImageGrid(x, res_h=self.res, res_w=self.res).reshape(1,-1)
        scale = res//self.res

        img = img.resize_by_scale(scale, method=Image.NEAREST)

        return img

    def eval(self):
        self.net.eval()

    def train(self):
        self.net.train()
