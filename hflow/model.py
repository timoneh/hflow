import os
import shutil
import time
from pathlib import Path
from argparse import Namespace
from copy import deepcopy
import socket


import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DPP



from hflow.dataloader import ImageDataset
import hflow.network as network
from hflow.utils import Network

from hflow.visualization import *

from hflow.metrics import compute_features_generator, compute_features_real, compute_fid, try_load_inception_net




class Model(object):

    def __init__(self, settings, skip_dataset=False, is_distributed=False):
        self.settings = settings

        self.result_path = self._prepare_res_and_checkpoint()

        if not skip_dataset:
            self.dev = torch.device("cuda:0")
            self.dataset, self.dataset_eval = ImageDataset.from_settings(self.settings)

            self.dataset_eval.h_flip = False
            self.dataset_eval.add_noise = False
            np.random.seed(self.settings["GENERAL"].get("SEED", 0))
            torch.manual_seed(self.settings["GENERAL"].get("SEED", 0))

            self.iterator = DataLoader(self.dataset, self.settings.batch_size, shuffle=True, num_workers=2, pin_memory=True)
            self.iterator_eval = DataLoader(self.dataset_eval, self.settings.batch_size, shuffle=True, num_workers=2)


            self.visualizations = [UnconditionalSample] 
            if not is_distributed:
                self.writer = SummaryWriter("tb")
            self.res = self.dataset.res

        else:
            self.res = None


        ns = self.settings.network


        mm = network.MultiStep(**ns)
        self.compute_metrics = True


        if not is_distributed:

            load_cp = self.settings["GENERAL"].get("LOAD_CHECKPOINT", None)


            if load_cp is None:
                encoder_params = mm.encoder_parameters()
                decoder_params = mm.decoder_parameters()
                flow_params = mm.non_encoder_parameters()

                opt = torch.optim.Adam(
                    [{"params": encoder_params, "lr":self.settings["GENERAL"]["LEARNING_RATE_ENCODERS"]},
                        {"params": flow_params},
                        {"params": decoder_params, "lr":self.settings["GENERAL"]["LEARNING_RATE_DECODERS"]}
                    ]
                    , lr=self.settings["GENERAL"]["LEARNING_RATE"])

                self.network = Network(mm, opt, settings=self.settings)
                self.network.settings = self.settings
                self.ema = Network(deepcopy(mm), None, settings=self.settings)
                self.ema.opt = self.network.opt

            else:
                tt = os.listdir(load_cp)
                tt.sort()
                tt.sort(key=len)

                path = load_cp + "/" + tt[-1]
                print("Trying to load from: ", path)
                temp = Network.load(path, skip_opt=True, force_cpu=True, dump_code=False)

                mm = temp.torch_network
                print("Loaded network from: ", path)

                encoder_params = mm.encoder_parameters()
                decoder_params = mm.decoder_parameters()
                flow_params = mm.non_encoder_parameters()

                opt = torch.optim.Adam(
                    [{"params": encoder_params, "lr":self.settings["GENERAL"]["LEARNING_RATE_ENCODERS"]},
                        {"params": flow_params},
                        {"params": decoder_params, "lr":self.settings["GENERAL"]["LEARNING_RATE_DECODERS"]}
                    ]
                    , lr=self.settings["GENERAL"]["LEARNING_RATE"])

                self.network = Network(temp.torch_network, opt, settings=self.settings)
                self.network.source = temp.source
                self.network.settings = self.settings
                self.ema = Network(deepcopy(self.network.torch_network), None, settings=self.settings)
                self.ema.opt = self.network.opt
                self.ema.source = self.network.source
        else:
            self.torch_network = mm

            load_cp = self.settings["GENERAL"].get("LOAD_CHECKPOINT", None)
            if load_cp is not None:
                del self.torch_network
                tt = os.listdir(load_cp)
                tt.sort()
                tt.sort(key=len)

                path = load_cp + "/" + tt[-1]
                print("Trying to load from: ", path)
                temp = Network.load(path, skip_opt=True, dump_code=False)

                self.torch_network = deepcopy(temp.torch_network)
                self.ema.source = self.network.source
                print("Loaded network from: ", path)




    def update_ema(self, a_w=0.01):
        with torch.no_grad():
            for a,b in zip(self.network.torch_network.parameters(), self.ema.torch_network.parameters()):
                b.copy_(b.lerp(a, a_w))


    def _prepare_res_and_checkpoint(self):
        path = Path.cwd()
        dirs = os.listdir(str(path))
        if "checkpoints" in dirs:
            shutil.rmtree(str(path/"checkpoints"))
        os.makedirs("checkpoints", exist_ok=True)
        if "results" in dirs:
            shutil.rmtree(str(path/"results"))
        os.makedirs("results", exist_ok=True) 
        if "tb" in dirs:
            shutil.rmtree(str(path/"tb"))
        os.makedirs("tb", exist_ok=True)
        if "cache" in dirs:
            shutil.rmtree(str(path/"cache"))
        os.makedirs("cache", exist_ok=True)
        return path/"results"

    @staticmethod
    def num_parameters(nw):
        return sum([item.numel() for item in nw.parameters()])



    def run_metrics(self, sampling_mod=1.0):

        bs = 32
        net_path = self.settings["GENERAL"]["INCEPTION_NET_PATH"]
        fid_sample_std = self.settings["OUTPUT"]["FID_GEN_STD"]
        n_samples = self.settings["OUTPUT"]["FID_N_SAMPLES"]

        inception = try_load_inception_net(net_path)
        if inception is None:
            return -1.0

        self.dataset.h_flip = False # to retain the original dataset
        dataset_disc = self.dataset._disc
        self.dataset.set_and_compute_disc(8) #compute with 8-bit data

        iterator = DataLoader(self.dataset, bs, shuffle=False, num_workers=0)

  
        print("Computing FID with {} samples".format(n_samples))
        features_gen = compute_features_generator(inception, self.ema.torch_network, bs, n_samples=n_samples, sample_std=fid_sample_std*sampling_mod)
        features_real = compute_features_real(inception,  iterator, tot_shown_max=n_samples)

        fid = compute_fid(features_real, features_gen, max_samples=n_samples)

        self.dataset.h_flip = True # back to training mode
        self.dataset.set_and_compute_disc(dataset_disc) #revert

        return fid


    @staticmethod
    def run_metrics_fn(settings, dataset, ema, sampling_mod=1.0):

        bs = 32
        net_path = settings["GENERAL"]["INCEPTION_NET_PATH"]
        fid_sample_std = settings["OUTPUT"]["FID_GEN_STD"]
        n_samples = settings["OUTPUT"]["FID_N_SAMPLES"]

        inception = try_load_inception_net(net_path)
        if inception is None:
            return -1.0

        dataset.h_flip = False # to retain the original dataset
        dataset_disc = dataset._disc
        dataset.set_and_compute_disc(8) #compute with 8-bit data

        iterator = DataLoader(dataset, bs, shuffle=False, num_workers=0)

        print("Computing FID with {} samples".format(n_samples))
        features_gen = compute_features_generator(inception, ema, bs, n_samples=n_samples, sample_std=fid_sample_std*sampling_mod)
        features_real = compute_features_real(inception, iterator, tot_shown_max=n_samples)

        dataset.h_flip = True # back to training mode
        dataset.set_and_compute_disc(dataset_disc) #revert

        fid = compute_fid(features_real, features_gen, max_samples=n_samples)
        return fid



    def run_visualizations(self, iteration):
        outs = []

        for vis_class in self.visualizations:
            vis = vis_class(self.settings, self.result_path, iteration)

            out = vis.generate(self.ema.torch_network, "_ema")
            vis.save()
            if out is not None:
                outs.append(out)
        return outs

    @staticmethod
    def visualize_distributed_fn(ema, model_ref, iteration):

        outs = []

        for vis_class in model_ref.visualizations:
            vis = vis_class(model_ref.settings, model_ref.result_path, iteration)

            out = vis.generate(ema, "_ema")
            vis.save()
            if out is not None:
                outs.append(out)
        return outs

    @staticmethod
    def train_distributed_fn(gpu_ind, args):
        rank = gpu_ind
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=args.world_size,
            rank=rank
        )

        settings = args.parent.settings
        np.random.seed(settings["GENERAL"].get("SEED", 0)*args.world_size + rank)
        torch.manual_seed(settings["GENERAL"].get("SEED", 0)*args.world_size + rank)


        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        dev = torch.device("cuda:{}".format(rank))
        ds, ds_test = ImageDataset.from_settings(settings)


        sampler = torch.utils.data.distributed.DistributedSampler(
            ds,
            num_replicas=args.world_size,
            rank=rank,
            drop_last=True
        )

        iterator = torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=settings["GENERAL"]["BATCH_SIZE"],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            sampler=sampler
        )
        iterator_eval = DataLoader(ds_test, settings.batch_size, shuffle=True, num_workers=0)

        mm = deepcopy(args.parent.torch_network).to(dev)
        mm.set_device(rank)

        if gpu_ind == 0:
            tot_params = Model.num_parameters(mm)
            print("Num Params {} M".format(tot_params*1e-6))


        model = DPP(mm, device_ids=[gpu_ind])



        encoder_params = []
        flow_params = []
        decoder_params = []

        for key,param in model.named_parameters():
            if "conv_d" not in key and not "conv_e" in key:
                flow_params.append(param)
            elif "conv_d" in key:
                decoder_params.append(param)
            else:
                encoder_params.append(param)
        opt = torch.optim.Adam(
                [{"params": encoder_params, "lr":settings["GENERAL"]["LEARNING_RATE_ENCODERS"]},
                {"params": flow_params},
                {"params": decoder_params, "lr":settings["GENERAL"]["LEARNING_RATE_DECODERS"]}
                ]
                , lr=settings["GENERAL"]["LEARNING_RATE"])

        dec_enc = settings["GENERAL"]["LEARNING_RATE_DECAY_ENCODERS"]
        dec = settings["GENERAL"]["LEARNING_RATE_DECAY"]
        enc_cutoff = settings["GENERAL"]["LEARNING_RATE_CUT_ENCODERS"]
        lmbd1 = lambda x: 0.0 if x>=enc_cutoff else dec_enc if (x%2==0 and x!=0) else 1.0
        lmbd2 = lambda x: dec if (x%2==0 and x!=0) else 1.0
        #(encoders, flows)
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=[lmbd1, lmbd2, lmbd2], verbose=True) #using lmbd2 for both decoders and flows

        sync_every = settings["GENERAL"]["SAMPLES_PER_GRAD_EVAL"]/(args.world_size*settings.batch_size)
        grads_scaled = float(1.0)/float(sync_every)

        if gpu_ind == 0:
            writer = SummaryWriter("tb")
            ema = deepcopy(model.module)

        def step(x, iter):


            if iter%sync_every == 0:

                ll = model(x)
                ll = torch.mean(ll)*grads_scaled
                ll_cpu = ll.detach().cpu().numpy()
                ll.backward()

                if settings["GENERAL"].get("GRAD_CLIP", 100.0) > 0.0:
                    torch.nn.utils.clip_grad_norm(model.parameters(), settings["GENERAL"].get("GRAD_CLIP", 100.0))

                opt.step()
                opt.zero_grad()
                t_sync = time.time()
                return ll_cpu/grads_scaled, t_sync
            else:
                with model.no_sync():
                    ll = model(x)
                    ll = torch.mean(ll)*grads_scaled
                    ll_cpu = ll.detach().cpu().numpy()
                    ll.backward()
                    return ll_cpu/grads_scaled, None

        def update_ema(a_w=0.01):
            for a,b in zip(model.parameters(), ema.parameters()):
                b.data = (1.0-a_w)*b.data + a_w*a.data



        tot_iters = 0
        tot_images_shown = 0
        t0_vis = time.time()
        t0_check = time.time()
        t0_metric = time.time()

        vis_target = settings["OUTPUT"]["VISUALIZATION_EVERY"]
        metric_target = settings["OUTPUT"]["METRICS_EVERY"]

        check_target = 0.3
        check_target_after_init = settings["OUTPUT"]["CHECKPOINT_EVERY"]
        check_target = min(check_target, check_target_after_init)
        t_sync = time.time()

        for e in range(settings.num_epochs):
            for ind, sample in enumerate(iterator):

                tot_iters += 1
                x = sample[0].to(dev)

                loss, t_sync_new = step(x, tot_iters)
                loss = loss  + ds.disc()

                if gpu_ind == 0 and tot_iters%sync_every == 0:
                    if e < 1:
                        update_ema(1.0)
                    else:
                        update_ema(0.005)


                tot_images_shown += settings.batch_size*args.world_size

                if tot_iters%sync_every == 0:
                    dt = t_sync_new - t_sync
                    t_sync = t_sync_new
                    didt = float(settings["GENERAL"]["SAMPLES_PER_GRAD_EVAL"])/dt

                    if gpu_ind == 0:
                        print("kImgs / images/s / gradEvals/s / loss: -- {:.2f} / {:.2f} / {:.2f} / {:.4f}".format(tot_images_shown/1000, didt, 1.0/dt, loss))


                t_current = time.time()
                t_vis_in_hours = (t_current - t0_vis)/3600.0
                t_check_in_hours = (t_current - t0_check)/3600.0
                t_metric_in_hours = (t_current - t0_metric)/3600.0


                if gpu_ind == 0:
                    if t_vis_in_hours > vis_target:
                        model.eval()
                        ema.eval()
                        with torch.no_grad():
                            try:
                                _ = Model.visualize_distributed_fn(ema, args.parent, int(tot_images_shown/1000.0))
                            except OSError:
                                print("Error saving visualizations, check permissions or disk-space")

                            tot_test_loss_ema = 0.0
                            norm = 0.0


                            for sample in iterator_eval:
                                x = sample[0].to(dev)
                                tot_test_loss_ema += torch.mean(ema.loss(x)).detach().cpu().numpy() + ds.disc()
                                norm += 1.0


                            layerwise = {}
                            for sample in iterator_eval:
                                x = sample[0].to(dev)
                                _ = ema.loss(x, return_layerwise=layerwise) # already numpy
                            

                            t = layerwise["ll_unscaled"]
                            ll_norm = layerwise["normalization"]
                            t = np.concatenate(t, axis=0)
                            t = np.mean(t)/(ll_norm*np.log(2.0)) + ds.disc()
                            writer.add_scalar("likelihood_bound/bpd_eval_ema", t, tot_images_shown)

                            writer.add_scalar("loss/eval_ema", tot_test_loss_ema/norm, tot_images_shown)
                            for ind,pg in enumerate(opt.param_groups):
                                lr = pg["lr"]
                                a = {2: "decoders", 1:"flows", 0:"encoders"}
                                writer.add_scalar("lr/{}".format(a[ind]), lr, tot_images_shown)



                            model.train()
                            ema.train()
                            t0_vis = time.time()
                            print("Ran visualizations!")
                                            
                    if t_check_in_hours > check_target:
                        temp = Network(ema, deepcopy(opt), settings=settings)
                        try:
                            temp.save("checkpoints/model_iter_{}.bin".format(tot_iters))
                            print("Saved checkpoint!")
                        except OSError:
                            print("Error saving checkpoint, likely met quota")
                        check_target = check_target_after_init # first checkpoint very early, change to longer intervals after
                        t0_check = time.time()


                    if t_metric_in_hours > metric_target:
                        ema.eval()
                        fid = Model.run_metrics_fn(settings, ds, ema)
                        ema.train()
                        writer.add_scalar("fid/fid", fid, tot_images_shown)
                        t0_metric = time.time()

            scheduler.step()


    def _port_in_use(self, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("localhost", port))
            s.close()
            return False
        except socket.error as e:
            return True



    def train_distributed(self):
        n_gpus = self.settings["GENERAL"]["NUM_GPUS"]

        print("Trying to use {} GPUs".format(n_gpus))

        print("Data-dependent inits before distribution...")
        tt = []
        N = 16
        for ind,sample in enumerate(self.iterator):
            if ind < N:
                tt.append(sample[0])
            else:
                break
        tt = torch.cat(tt, dim=0)
        load_cp = self.settings["GENERAL"].get("LOAD_CHECKPOINT", None)
        if load_cp is None:
            with torch.no_grad():
                _ = self.torch_network.loss(tt)
            print("Done")



        args = Namespace()
        args.world_size = n_gpus
        args.parent = self
        
        BASE_PORT = 8008
        while self._port_in_use(BASE_PORT):
            BASE_PORT = BASE_PORT + 1
        
        print("Using port {} for multiprocessing".format(BASE_PORT))
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "{}".format(BASE_PORT)

        torch.multiprocessing.set_sharing_strategy("file_system")
        torch.multiprocessing.spawn(Model.train_distributed_fn, nprocs=n_gpus, args=(args,))


    def train(self):

        tot_iters = 0
        self.network.torch_network = self.network.torch_network.to(self.dev)
        self.ema.torch_network = self.ema.torch_network.to(self.dev)

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


        np.random.seed(self.settings["GENERAL"].get("SEED", 0))
        torch.manual_seed(self.settings["GENERAL"].get("SEED", 0))

        tot_params = self.num_parameters(self.network.torch_network)
        print("Num Params {} M".format(tot_params*1e-6))

        tot_images_shown = 0
        t0_vis = time.time()
        t0_check = time.time()
        t0_metric = time.time()

        vis_target = self.settings["OUTPUT"]["VISUALIZATION_EVERY"]
        metric_target = self.settings["OUTPUT"]["METRICS_EVERY"]

        check_target = 0.03
        check_target_after_init = self.settings["OUTPUT"]["CHECKPOINT_EVERY"]
        check_target = min(check_target, check_target_after_init)


        tt = []
        for ind,sample in enumerate(self.iterator):
            if ind < 16:
                tt.append(sample[0])
            else:
                break
        tt = torch.cat(tt, dim=0).cuda()
        with torch.no_grad():
            _ = self.network.torch_network.loss(tt)

        self.network.opt.zero_grad()
        for e in range(self.settings.num_epochs):



            for ind, sample in enumerate(self.iterator):

                tot_iters += 1
                x = sample[0].to(self.dev)
                t = time.time()


                every = self.settings["GENERAL"]["SAMPLES_PER_GRAD_EVAL"]//self.settings.batch_size
                step = ind%every == 0
                grads_per_batch = float(1.0)/float(every)

                loss = self.network.step(x, zero_and_step=step, scale=grads_per_batch) + self.dataset.disc()
                if step:
                    if e < 1:
                        self.update_ema(1.0)
                    else:
                        self.update_ema(0.005)


                dt = time.time() - t
                didt = float(self.settings.batch_size)/dt

                tot_images_shown += self.settings.batch_size
                print("kImgs / images/s / gradEvals/s / loss: -- {:.2f} / {:.2f} / {:.2f} / {:.4f}".format(tot_images_shown/1000, didt, grads_per_batch/dt, loss))


                t_current = time.time()
                t_vis_in_hours = (t_current - t0_vis)/3600.0
                t_check_in_hours = (t_current - t0_check)/3600.0
                t_metric_in_hours = (t_current - t0_metric)/3600.0

                if t_vis_in_hours > vis_target:
                    print("Running visualizations...")
                    self.network.torch_network.eval()
                    self.ema.torch_network.eval()
                    with torch.no_grad():
                        try:
                            _ = self.run_visualizations(int(tot_images_shown/1000.0))
                        except OSError as e:
                            print("Error saving visualizations, check permissions or disk-space")

                        tot_test_loss = 0.0
                        tot_test_loss_ema = 0.0
                        norm = 0.0


                        for sample in self.iterator_eval:
                            x = sample[0].to(self.dev)
                            tot_test_loss += torch.mean(self.network.torch_network.loss(x)).detach().cpu().numpy() + self.dataset.disc()
                            tot_test_loss_ema += torch.mean(self.ema.torch_network.loss(x)).detach().cpu().numpy() + self.dataset.disc()
                            norm += 1.0



                        layerwise = {}
                        for sample in self.iterator_eval:
                            x = sample[0].to(self.dev)
                            _ = self.ema.torch_network.loss(x, return_layerwise=layerwise) # already numpy
                        


                        t = layerwise["ll_unscaled"]
                        ll_norm = layerwise["normalization"]
                        t = np.concatenate(t, axis=0)
                        t = np.mean(t)/(ll_norm*np.log(2.0)) + self.dataset.disc()
                        self.writer.add_scalar("likelihood_bound/bpd_eval_ema", t, tot_images_shown)

                        self.writer.add_scalar("loss/eval_ema", tot_test_loss_ema/norm, tot_images_shown)


                        for ind,pg in enumerate(self.network.opt.param_groups):
                            lr = pg["lr"]
                            a = {2: "decoders", 1:"flows", 0:"encoders"}
                            self.writer.add_scalar("lr/{}".format(a[ind]), lr, tot_images_shown)


                        self.network.torch_network.train()
                        self.ema.torch_network.train()
                        t0_vis = time.time()
                        print("Ran visualizations!")
                                        
                if t_check_in_hours > check_target:
                    try:
                        self.ema.save("checkpoints/model_iter_{}.bin".format(tot_iters))
                        print("Saved checkpoint!")
                    except OSError as err:
                        print("Error saving checkpoint, likely met quota")
                    check_target = check_target_after_init # first checkpoint very early, change to longer intervals after
                    t0_check = time.time()

                if t_metric_in_hours > metric_target:
                    self.ema.torch_network.eval()
                    fid = self.run_metrics()
                    self.ema.torch_network.train()

                    self.writer.add_scalar("fid/fid", fid, tot_images_shown)
                    t0_metric = time.time()

            self.network.scheduler.step()

