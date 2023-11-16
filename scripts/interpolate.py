import hflow
import torch
import argparse


def interpolate(model, img_a, img_b, steps, fps):
    with torch.no_grad():
        img_a = model.load_img(img_a)
        img_b = model.load_img(img_b)
        latent_a = model.inverse(img_b)
        latent_b = model.inverse(img_a)
        out = model.simple_interpolation(None, None, steps, override_a_with_z=latent_a, override_b_with_z=latent_b)
        out.toVideo((1,1), mirror=True, fps=fps, filename="interpolation.mp4")




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to model file to be loaded")
    parser.add_argument("-p1", "--path_to_img1", type=str, required=True, help="Path to image interpolation start-point")
    parser.add_argument("-p2", "--path_to_img2", type=str, required=True, help="Path to image interpolation end-point")
    parser.add_argument("-N", "--n_steps", type=int, default=48, help="Path to image interpolation end-point")
    parser.add_argument("-FPS", "--frames_per_second", type=int, default=24, help="Path to image interpolation end-point")


    args = parser.parse_args()
    model = hflow.Wrapper.from_path(args.path)
    model.eval()
    model.net.set_noise_active(False)

    interpolate(model, args.path_to_img1, args.path_to_img2, args.n_steps, args.frames_per_second)
