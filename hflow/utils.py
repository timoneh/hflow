import inspect
import tempfile
import os
import json
import sys
import importlib.util

import torch

import numpy as np
import h5py
import io
import zipfile
from PIL import Image


# Wrapper around a dict for accessing elements with .-operation
# adapted from https://github.com/NVlabs/stylegan3/blob/main/dnnlib/util.py
class EasyDict(dict):

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


class ModelSettings(object):

    def __init__(self, in_dict):
        self.raw_dict = in_dict

        self.general = self._toInternal(in_dict["GENERAL"])
        self.network = self._toInternal(in_dict["MODEL_PARAMETERS"])


    def _toInternal(self, in_dict):
        temp = {}
        for key,value in in_dict.items():
            if type(value) is dict:
                temp[key.lower()] = self._toInternal(value)
            else:
                temp[key.lower()] = value
        return EasyDict(temp)


    def __getitem__(self, key):
        return self.raw_dict[key]

    def toString(self):
        ds = self.raw_dict["GENERAL"]["DATASET"]
        res = self.raw_dict["GENERAL"]["RESIZE_DATA_TO"]
        return ds + "_" + str(res)
   

    @property
    def batch_size(self):
        return self.general.batch_size

    @property
    def num_epochs(self):
        return self.general.epochs


    @classmethod
    def from_file(cls, path):
        if type(path) is str:
            with open(path, "r") as f:
                js = json.loads(f.read())
            return cls(js)
        else:
            with path.open() as f:
                js = json.loads(f.read())
            return cls(js)

    @staticmethod
    def extract_json(path):
        if type(path) is str:
            with open(path, "r") as f:
                js = json.loads(f.read())
            return js
        else:
            with path.open() as f:
                js = json.loads(f.read())
            return js



class Network(object):

    def __init__(self, torch_network, opt, l_rate=1e-5, settings=None, force_cpu=False):
        if not force_cpu:
            self.torch_network = torch_network.cuda()
        else:
            self.torch_network = torch_network
        self.opt = opt
        self.settings = settings
        dec_enc = 0.85 if self.settings is None else self.settings["GENERAL"]["LEARNING_RATE_DECAY_ENCODERS"]
        dec = 0.95 if self.settings is None else self.settings["GENERAL"]["LEARNING_RATE_DECAY"]
        dec_start = 20 if self.settings is None else self.settings["GENERAL"]["LEARNING_RATE_DECAY_START"]
        enc_cutoff = self.settings["GENERAL"]["LEARNING_RATE_CUT_ENCODERS"] if settings is not None else 1
        if opt is not None:
            lmbd1 = lambda x: 0.0 if x>=enc_cutoff else dec_enc if (x%2==0 and x!=0 and x > dec_start) else 1.0
            lmbd2 = lambda x: dec if (x%2==0 and x!=0 and x > dec_start) else 1.0
            #(encoders, flows)
            try:
                self.scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.opt, lr_lambda=[lmbd1, lmbd2, lmbd2], verbose=True) #using lmbd2 for both decoders and flows
            except ValueError:
                self.scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.opt, lr_lambda=[lmbd2], verbose=True) #using lmbd2 for both decoders and flows


        self.source = None



    def save(self, path="model.bin"):

        module = inspect.getmodule(self.torch_network)

        if self.source is None:
            src = inspect.getsource(module)
        else:
            src = self.source

        mod_name = str(module.__name__).encode("utf-8")
        mod_name_len = len(mod_name)
        src_utf = src.encode("utf-8")
        src_len = len(src_utf)

        settings_dict = self.settings.raw_dict
        settings_dict_str = json.dumps(settings_dict)
        settings_dict_str_len = len(settings_dict_str)
        settings_dict_str_utf = settings_dict_str.encode("utf-8")

        with open(path, mode="wb") as f:
            f.write(mod_name_len.to_bytes(4, byteorder="little", signed=False))
            f.write(mod_name)
            f.write(src_len.to_bytes(4, byteorder="little", signed=False))
            f.write(src_utf)


            torch.save(self.torch_network, f, _use_new_zipfile_serialization=False)
            torch.save(self.opt.state_dict(), f, _use_new_zipfile_serialization=False)

            f.write(settings_dict_str_len.to_bytes(4, byteorder="little", signed=False))
            f.write(settings_dict_str_utf)



    def step(self, x, zero_and_step=True, scale=1.0):


        loss = self.torch_network.loss(x)
        loss = torch.mean(loss)*scale #to scale gradients if we are accumulating over multiple batches
        loss_cpu = loss.detach().cpu().numpy()


        loss.backward()

        if zero_and_step:
            if self.settings["GENERAL"].get("GRAD_CLIP", -1.0) > 0.0:
                torch.nn.utils.clip_grad_norm(self.torch_network.parameters(), self.settings["GENERAL"].get("GRAD_CLIP", 100.0))
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

        return loss_cpu/scale


    def unused_params(self):
        for k,v in self.torch_network.named_parameters():
            if v.requires_grad and v.grad is None:
                print(k)


    @classmethod
    def load(cls, path="model.bin", skip_opt=False, force_cpu=False, dump_code=False):
        with open(path, "rb") as f:
            mod_name_len = int.from_bytes(f.read(4), byteorder="little", signed=False)
            mod_name = f.read(mod_name_len).decode()
            mod_name = "hflow.network"

            src_len = int.from_bytes(f.read(4), byteorder="little", signed=False)
            src = f.read(src_len).decode()


            if dump_code:
                with open("network.py", "w") as fd:
                    fd.write(src)

            mod = None
            if os.path.exists("tmp_hflow_network_code.py"):
                raise RuntimeError("When loading the network, tried to dump & load model code using file 'tmp_hflow_network_code.py' -- but it already existed -- remove it and try again. It could be leftover from a previous load. If it is you own file, move it and try again.")
            with open("tmp_hflow_network_code.py", "wb") as temp_f:

                mod_name = "hflow.network"

                temp_f.write(src.encode())
                temp_f.flush()

                spec = importlib.util.spec_from_file_location(mod_name, temp_f.name)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                sys.modules[mod.__name__] = mod
                nw = torch.load(f).cpu() if force_cpu else torch.load(f)
            os.remove("tmp_hflow_network_code.py")

            if not skip_opt:
                try:
                    out = cls(nw, torch.optim.Adam(nw.parameters(), lr=1e-5), force_cpu=force_cpu)
                    print("Set learning rate is a placeholder.")
                except ValueError: #there might be two (or more) parameter groups for newer models
                    encoder_params = nw.encoder_parameters()
                    decoder_params = nw.decoder_parameters()
                    flow_params = nw.non_encoder_parameters()
                    opt = torch.optim.Adam(
                            [{"params": encoder_params},
                            {"params": flow_params},
                            {"params": decoder_params}
                            ]
                            , lr=1e-5)
                    out = cls(nw, opt, force_cpu=force_cpu)
            else:
                opt = None
                out = cls(nw, opt, force_cpu=force_cpu)

            state_d = torch.load(f)
            if not skip_opt:
                try:
                    out.opt.load_state_dict(state_d)
                except ValueError:
                    pass
            out.source = src

            settings_len = f.read(4)
            if settings_len:
                settings_len = int.from_bytes(settings_len, byteorder="little", signed=False)
                settings_read = f.read(settings_len).decode()
                settings_read = json.loads(settings_read)
                out.settings = ModelSettings(settings_read)
            else: #was EOF, model had no settings-dict saved
                out.settings = None

            return out



def dataset_from_zipfile(path_to_zipfile, hdf5_filepath, dataset_name, target_resolution):
    """Generate a dataset to be used in training from a zip of images
    args:
        path_to_zipfile: path to images
        hdf5_filepath: path where the dataset will be created
        dataset_name: name of the dataset. Will need to be specified to the training code.
        target_resolution: images will be reshaped to this resolution using PIL's resize-function with the Lanczos filter
    out:
        None, as a side-effect, a hdf5 file is created to hdf5_filepath
    """


    def generator():
        with zipfile.ZipFile(path_to_zipfile, "r") as f:
            filenames = list(filter(lambda x: ".png" in x, f.namelist()))
            if not filenames:
                raise RuntimeError("No png-files found in the zipfile.")
            for img_name in filenames:
                data = f.read(img_name)
                temp = io.BytesIO(data)
                with Image.open(temp) as img:
                    t = img.resize((target_resolution, target_resolution), Image.Resampling.LANCZOS)
                    t_data = np.asarray(t).astype(np.uint8)
                    yield t_data



    with zipfile.ZipFile(path_to_zipfile, "r") as f:
        tot_imgs = len(list(filter(lambda x: ".png" in x, f.namelist())))
    tot_shape = [tot_imgs, target_resolution, target_resolution, 3]

    with h5py.File(hdf5_filepath, "a") as hfile:
        keys = hfile.keys()
        if not dataset_name in keys:
            group = hfile.create_group(dataset_name)
            ds = group.create_dataset("X_DATA", tot_shape, dtype="uint8")
        else:
            group = hfile[dataset_name]
            ds = group["X_DATA"]

        ind = 0
        for img in generator():
            ds[ind:ind+1,...] = img
            ind = ind + 1


