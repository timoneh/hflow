import torch
import numpy as np
import scipy.linalg as la
import os
import pickle



def try_load_inception_net(path, hard_fail=False):
    try:
        with open(path, "rb") as f:
            try:
                inception = pickle.load(f).cuda()
                return inception
            except ModuleNotFoundError as e:
                if hard_fail:
                    raise e
                print("Failed to load inception-net due to missing stylegan3-related modules. Is the module available and on the PYTHONPATH?")
                print("Skipping FID-computation.")
                return None
    except FileNotFoundError as e:
        if hard_fail:
            raise e
        print("Inception-net not found at: {}".format(path))
        print("Skipping FID-computation.")
        return None



def compute_features_real(model, dataset_iterator, tot_shown_max=25000, cache_path_override=None, noisy=True, verbose=True, save_features=True):

    out = []

    cache_path = "cache/fid_features_reals{}.npy".format("_clean" if not noisy else "") if cache_path_override is None else cache_path_override
    is_cached = False
    tot_shown = 0

    if os.path.exists(cache_path) and os.path.isfile(cache_path):
        out = np.load(cache_path)
        is_cached = True
        if verbose:
            print("Using precomputed features from: ", cache_path)
    else:
        print("Features not found from cache path: ", cache_path)
        with torch.no_grad():
            for N,item in enumerate(dataset_iterator):
                if tot_shown > tot_shown_max:
                    break
                if N%10 == 0 and verbose:
                    print("FID // real - Progress: {:.2f}".format(tot_shown/tot_shown_max * 100.0))
                item = item[0].cuda() + 0.5 # our data is [-0.5, 0.5]
                item = item.clamp(0.0, 1.0)*255.0
                item = item.to(torch.uint8)
                y = model(item, return_features=True)
                tot_shown = tot_shown + y.shape[0]
                out.append(y.detach().cpu().numpy())
        out = np.concatenate(out, axis=0)
    if not is_cached and save_features:
        np.save(cache_path, out)


    out = out[:tot_shown_max]
    if verbose:
        print("Real shape,", out.shape)

    return out.squeeze()



def sample_normalized(generator, base, std_list):


    tail = base[-1]
    head = base[:-1]
    tail = list(generator.endblock_unified.toInternal(tail))
    if len(tail) == 2 and tail[-1] is None:
        tail = [tail[0]]

    base = head + tail

    Z = list(map(lambda x: torch.randn_like(x[0])*x[1], zip(base, std_list)))


    nn = len(generator.endblock_unified.c)
    tail = Z[-nn:]
    head = Z[:-nn]

    tail = [generator.endblock_unified.fromInternal(tail)]
    Z = head + tail




    t,_ = generator.flow_forward(Z)
    t = (t + 0.5).clamp(0.0, 1.0)
    return t


def compute_features_generator(model, generator, bs, n_samples, sample_std=1.0, verbose=True):

    n_batches = n_samples//bs + 1
    out = []
    simple_std = True

    if type(sample_std) is list:
        with torch.no_grad():
            base = generator._sample(bs, std=1.0)
            base,_ = generator.inverse(base)
        simple_std = False


    with torch.no_grad():
        for N in range(n_batches):
            if N%10 == 0 and verbose:
                print("FID // generated - Progress: {:.2f}".format(N/n_batches * 100.0))
            if simple_std:
                samples = generator.sample_normalized(bs, std=sample_std)*255.0
            else:
                samples = sample_normalized(generator, base, std_list=sample_std)*255.0
            samples = samples.clamp(0.0, 255.0)
            samples = samples.to(torch.uint8)
            y = model(samples, return_features=True)
            out.append(y.detach().cpu().numpy())

    out = np.concatenate(out, axis=0)

    out = out[:n_samples]
    if verbose:
        print("Gen shape,", out.shape)

    return out.squeeze()


def compute_fid(features_real, features_gen, max_samples=25000):

    features_real = features_real[0:max_samples,...].astype(np.float64)
    features_gen = features_gen[0:max_samples,...].astype(np.float64)

    mu_real = np.mean(features_real, axis=0)
    mu_gen = np.mean(features_gen, axis=0)

    sigma_real = np.cov(features_real.T)
    sigma_gen = np.cov(features_gen.T)


    sq,_ = la.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)

    diff = np.sum(np.square(mu_real-mu_gen))
    a = np.real(diff  + np.trace(sigma_gen + sigma_real - 2*sq))

    return a
