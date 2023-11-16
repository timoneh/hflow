from hflow import dataset_from_zipfile
import torch
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to zipfile containing images")
    parser.add_argument("-dp", "--dataset_path", type=str, required=True, help="Path to created dataset")
    parser.add_argument("-dn", "--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("-r", "--resolution", type=int, required=True, help="Resolution of the dataset")


    args = parser.parse_args()
    dataset_from_zipfile(args.path, args.dataset_path, args.dataset_name, args.resolution)


