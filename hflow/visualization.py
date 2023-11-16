import torch
import matplotlib.pyplot as plt
from hflow.imagegrid import ImageGrid
from hflow.wrapper import Wrapper



class Visualization(object):

    def __init__(self, settings, path, i):
        self.path = path
        self.settings = settings
        self.iter = i
        self.imgs = []
        self.notes = []
        self.name = "undef"

    def generate(self, network):
        raise NotImplementedError()

    def save(self):
        for ind, (img, note) in enumerate(zip(self.imgs, self.notes)):
            name = "kimg_{}_ind_{}{}_{}.png".format(self.iter, ind, note, self.name)
            if type(img) is ImageGrid:
                img.save(str(self.path/name), quality=100)
            else:
                img.savefig(str(self.path/name))
                plt.close(img)


class UnconditionalSample(Visualization):

    def __init__(self, settings, path, i):
        super(UnconditionalSample, self).__init__(settings, path, i)
        self.name = "unconditional_sample"

    def generate(self, network, notes=""):

        N = 12 if network.fullres < 256 else 6


        network.eval()

        with torch.no_grad():
            stds = [1.0] + [self.settings["OUTPUT"]["SAMPLE_BASE_SIGMA"]*t for t in [1.0,0.8,0.5,0.3]]
            X = list(map(lambda x: Wrapper.to_image(network.sample(N, std=x)), stds))
            img = ImageGrid.stackH(X)

        network.train()



        img = img.pad(1,1)
        self.imgs.append(img)
        self.notes.append(notes)

