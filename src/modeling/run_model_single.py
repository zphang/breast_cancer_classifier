# Copyright (C) 2019 Nan Wu, Jason Phang, Jungkyu Park, Yiqiu Shen, Zhe Huang, Masha Zorin,
#   Stanisław Jastrzębski, Thibault Févry, Joe Katsnelson, Eric Kim, Stacey Wolfson, Ujas Parikh,
#   Sushma Gaddam, Leng Leng Young Lin, Kara Ho, Joshua D. Weinstein, Beatriu Reig, Yiming Gao,
#   Hildegard Toth, Kristine Pysarenko, Alana Lewin, Jiyon Lee, Krystal Airola, Eralda Mema,
#   Stephanie Chung, Esther Hwang, Naziya Samreen, S. Gene Kim, Laura Heacock, Linda Moy,
#   Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of breast_cancer_classifier.
#
# breast_cancer_classifier is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# breast_cancer_classifier is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with breast_cancer_classifier.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================
"""
Runs the image only model and image+heatmaps model for breast cancer prediction.
"""
import argparse
import collections as col
import numpy as np
import os
import pandas as pd
import torch
import json
import tqdm

import src.utilities.pickling as pickling
import src.utilities.tools as tools
import src.modeling.models as models
import src.data_loading.loading as loading
from src.constants import VIEWS, VIEWANGLES, LABELS, MODELMODES


def load_model():
    pass


def run(parameters):
    """
    Outputs the predictions as csv file
    """
    random_number_generator = np.random.RandomState(parameters["seed"])
    input_channels = 3 if parameters["use_heatmaps"] else 1
    model = models.SingleImageBreastModel(input_channels)
    model.load_state_dict(torch.load(parameters["model_path"])["model"])

    if (parameters["device_type"] == "gpu") and torch.has_cudnn:
        device = torch.device("cuda:{}".format(parameters["gpu_number"]))
    else:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    metadata = pickling.unpickle_from_file(parameters['metadata_path'])
    view = metadata["full_view"]

    image = loading.load_image(
        image_path=parameters["cropped_mammogram_path"],
        view=view,
        horizontal_flip=metadata["horizontal_flip"],
    )
    if parameters["use_heatmaps"]:
        loaded_heatmaps = loading.load_heatmaps(
            benign_heatmap_path=parameters["heatmap_path_benign"],
            malignant_heatmap_path=parameters["heatmap_path_malignant"],
            view=view,
            horizontal_flip=metadata["horizontal_flip"],
        )
    else:
        loaded_heatmaps = None

    all_predictions = []
    for data_batch in tools.partition_batch(range(parameters["num_epochs"]), parameters["batch_size"]):
        batch = []
        for _ in data_batch:
            cropped_image, cropped_heatmaps = loading.augment_and_normalize_image(
                image=image,
                auxiliary_image=loaded_heatmaps,
                view=view,
                best_center=metadata["best_center"],
                random_number_generator=random_number_generator,
                augmentation=parameters["augmentation"],
                max_crop_noise=parameters["max_crop_noise"],
                max_crop_size_noise=parameters["max_crop_size_noise"],
            )
            if parameters["use_heatmaps"]:
                batch.append(np.concatenate([
                    cropped_image[:, :, np.newaxis],
                    cropped_heatmaps,
                ], axis=2))
            else:
                batch.append(cropped_image[:, :, np.newaxis])
        tensor_batch = torch.tensor(np.stack(batch)).permute(0, 3, 1, 2).to(device)
        y_hat = model.single_forward(tensor_batch, view=view)
        predictions = np.exp(y_hat.cpu().detach().numpy())[:, :2, 1]
        all_predictions.append(predictions)
    agg_predictions = np.concatenate(all_predictions, axis=0).mean(0)
    print(json.dumps({
        "benign": float(agg_predictions[0]),
        "malignant": float(agg_predictions[1]),
    }))


def main():
    parser = argparse.ArgumentParser(description='Run image-only model or image+heatmap model')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--cropped-mammogram-path', required=True)
    parser.add_argument('--metadata-path', required=True)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--use-heatmaps', action="store_true")
    parser.add_argument('--heatmap-path-malignant')
    parser.add_argument('--heatmap-path-benign')
    parser.add_argument('--use-augmentation', action="store_true")
    parser.add_argument('--use-hdf5', action="store_true")
    parser.add_argument('--num-epochs', default=1, type=int)
    parser.add_argument('--device-type', default="cpu", choices=['gpu', 'cpu'])
    parser.add_argument("--gpu-number", type=int, default=0)
    args = parser.parse_args()

    parameters = {
        "model_path": args.model_path,
        "cropped_mammogram_path": args.cropped_mammogram_path,
        "metadata_path": args.metadata_path,
        "device_type": args.device_type,
        "gpu_number": args.gpu_number,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "augmentation": args.use_augmentation,
        "num_epochs": args.num_epochs,
        "use_heatmaps": args.use_heatmaps,
        "heatmap_path_benign": args.heatmap_path_benign,
        "heatmap_path_malignant": args.heatmap_path_malignant,
        "use_hdf5": args.use_hdf5,
    }
    run(parameters)


if __name__ == "__main__":
    main()

