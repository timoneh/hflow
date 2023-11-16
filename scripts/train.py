import hflow
import torch
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config-based")
    parser.add_argument("-dp", "--data_path", type=str, required=True, help="Path to hdf5 data-container")
    parser.add_argument("-dn", "--data_name", type=str, required=True, help="Name of dataset")
    parser.add_argument("-ip", "--inception_net_path", type=str, default="", help="Path to Inception-network pickle -- optionally used in FID computation.")
    parser.add_argument("-ie", "--images_every_hours", type=float, default=0.5, help="Output model samples every h hours")
    parser.add_argument("-me", "--metrics_every_hours", type=float, default=4.0, help="Compute metrics every h hours")
    parser.add_argument("-ce", "--checkpoint_every_hours", type=float, default=4.0, help="Dump model checkpoint every h hours")
    parser.add_argument("-ct", "--continue_folder", type=str, default=None, help="Path of a folder of checkpoints out of which the latest is used as the initialization for continuing training.")


    args = parser.parse_args()
    settings = hflow.ModelSettings.from_file(args.config)
    settings["GENERAL"]["DATASET_PATH"] = args.data_path
    settings["GENERAL"]["DATASET_NAME"] = args.data_name
    settings["GENERAL"]["INCEPTION_NET_PATH"] = args.inception_net_path
    settings["OUTPUT"]["VISUALIZATION_EVERY"] = args.images_every_hours
    settings["OUTPUT"]["METRICS_EVERY"] = args.metrics_every_hours
    settings["OUTPUT"]["CHECKPOINT_EVERY"] = args.checkpoint_every_hours
    settings["GENERAL"]["LOAD_CHECKPOINT"] = args.continue_folder

    print("Training...")
    print("Visualizations every {} hours / metrics every {} hours / checkpoints every {} hours.".format(settings["OUTPUT"]["VISUALIZATION_EVERY"],
                                                                                                        settings["OUTPUT"]["METRICS_EVERY"],
                                                                                                        settings["OUTPUT"]["CHECKPOINT_EVERY"]))
    print("#####")

    is_distributed = settings["GENERAL"]["NUM_GPUS"] > 1
    model = hflow.Model(settings, is_distributed=is_distributed)

    if args.inception_net_path:
        _ = hflow.try_load_inception_net(args.inception_net_path, hard_fail=True)
        print("Inception-net was found and succesfully loaded from:", args.inception_net_path)

    if is_distributed:
        model.train_distributed()
    else:
        model.train()


