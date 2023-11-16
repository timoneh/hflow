import numpy as np
from PIL import Image
import h5py
from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self, path="", ds="", discretisation_bits=8, const_rand_scale=0.01, noise_std=0.0, resize_to=None, data=None, name="unnamed_dataset", use_for_eval=200, is_trainset=True):
        super().__init__()


        self.dataset = None
        self.path = path
        self.ds_name = ds
        self.is_trainset = is_trainset
        self.use_for_eval = use_for_eval

        if data is not None:
            self.dataset = data[0]
            self.dataset_len = self.dataset.shape[0]
            self.shape = self.dataset.shape
            self.init_ind = 0
        else:
            with h5py.File(self.path, "r") as file:
                if self.is_trainset:
                    self.init_ind = 0
                    self.dataset_len = len(file[self.ds_name]["X_DATA"])
                    if self.dataset_len < 100000:
                        print("Trying to load train dataset to local memory -- caps at 100k imgs -- OOM possible if very large images.")
                        self.dataset = file[self.ds_name]["X_DATA"][()]
                    self.shape = file[self.ds_name]["X_DATA"].shape
                else:
                    self.init_ind = 0
                    self.use_for_eval = min(self.use_for_eval, int(len(file[self.ds_name]["X_DATA"])))
                    ds_start_ind = int(len(file[self.ds_name]["X_DATA"])) - self.use_for_eval
                    self.dataset_len = self.use_for_eval
                    print("Trying to load to local memory #items: ", self.dataset_len)
                    self.shape = file[self.ds_name]["X_DATA"].shape

                    self.dataset = np.copy(file[self.ds_name]["X_DATA"][ds_start_ind:])

        print("Dataset {} with shape {} with dtype {}".format(name, self.dataset.shape, self.dataset.dtype))
        if not self.is_trainset:
            print("Eval-fraction (subset of the trainset) was loaded to local for evaluating the model.")

        self.rand_scale = const_rand_scale
        self._disc = discretisation_bits
        self.divisor_int = 256 // 2**(self._disc)
        self.divisor_float = float(2**(self._disc))
        self.noise_std = noise_std
        self.resize_to = resize_to
        self.h_flip = True
        self.add_noise = True
        self.name = name



    def disc(self):
        return self._disc



    @classmethod
    def from_settings(cls, settings):
        path = settings["GENERAL"]["DATASET_PATH"]
        ds = settings["GENERAL"]["DATASET_NAME"]
        ns = settings["GENERAL"]["ADDED_NOISE_STD"]
        resize = settings["GENERAL"]["RESIZE_DATA_TO"]

        return cls(path=path, ds=ds, discretisation_bits=settings["GENERAL"]["DISCRETIZATION_BITS"], noise_std=ns, resize_to=resize, data=None, name=ds+"_train", is_trainset=True), \
               cls(path=path, ds=ds, discretisation_bits=settings["GENERAL"]["DISCRETIZATION_BITS"], noise_std=ns, resize_to=resize, data=None, name=ds+"_test", is_trainset=False)


    def set_and_compute_disc(self, bits):
        print("Set discretization to", bits)
        self._disc = bits
        self.divisor_int = 256 // 2**(self._disc)
        self.divisor_float = float(2**(self._disc))


    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):

        if self.dataset is None:
            self.dataset = h5py.File(self.path, "r")[self.ds_name]["X_DATA"]


        data = self.dataset[idx+self.init_ind]

        if self.resize_to is not None:
            temp_x = Image.fromarray(data)
            temp_x = temp_x.resize((self.resize_to, self.resize_to), Image.LANCZOS)
            temp_x  = np.asarray(temp_x).transpose(2,0,1)
        else:
            temp_x = self.dataset[idx+self.init_ind].transpose(2,0,1)# * scale


        
        temp_x = (temp_x // self.divisor_int).astype(np.float32)/self.divisor_float #dequantization


        if np.random.uniform() < 0.5 and self.h_flip:
            temp_x = np.flip(temp_x, axis=(-1))

        if self.add_noise:
            temp_x = temp_x + np.random.uniform(low=0.0, high=1.0/self.divisor_float, size=temp_x.shape)

        temp_x = temp_x - 0.5

        if self.noise_std > 0.0 and self.add_noise:
            noise = (np.random.normal(size=temp_x.shape)*self.noise_std)
            temp_x = temp_x + noise.astype(np.float32)
        
        out = temp_x.astype(np.float32)
        return out, out


    @property
    def res(self):
        return self.shape[-1]

    @property
    def channels(self):
        return self.shape[0]


