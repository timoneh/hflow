import hflow
import torch
import argparse


def sample(model, n_samples, sampling_std):
    with torch.no_grad():
        samples = model.sample(n_samples, sampling_std)
        samples = hflow.Wrapper.to_image(samples)
        samples.save("samples.png")




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to model file to be loaded")
    parser.add_argument("-N", "--n_samples", type=int, default=1, help="Number of samples")
    parser.add_argument("-s", "--sample_std", type=float, default=0.8, help="Number of samples")


    args = parser.parse_args()
    model = hflow.Wrapper.from_path(args.path)
    model.eval()

    sample(model, args.n_samples, args.sample_std)
