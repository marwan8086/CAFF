import argparse
import os
import yaml
import torch
import random
import numpy as np

from data.dataset import CAFFDataset
from models.caff_model import CAFFModel
from training.trainer import Trainer
from utils.helpers import set_seed, ensure_dir


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str):
    config = load_config(config_path)
    ensure_dir(config["save_dir"])

    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    train_dataset = CAFFDataset(
        json_path=config["train_json"],
        kg_path=config["merged_kg_path"],
        encoder_name=config["encoder_name"],
        max_relation_length=config["max_relation_length"],
        max_query_length=config["max_query_length"],
        L=config["L"],
        Kr=config["Kr"],
        device=device,
    )

    dev_dataset = CAFFDataset(
        json_path=config["dev_json"],
        kg_path=config["merged_kg_path"],
        encoder_name=config["encoder_name"],
        max_relation_length=config["max_relation_length"],
        max_query_length=config["max_query_length"],
        L=config["L"],
        Kr=config["Kr"],
        device=device,
    )

    model = CAFFModel(
        encoder_name=config["encoder_name"],
        d_model=config["encoder_dim"],
        rho=config["rho"],
        L=config["L"],
        num_relations=train_dataset.num_relations,
        device=device,
    ).to(device)

    model.set_relation_embeddings(train_dataset.relation_texts)
    train_dataset.set_relation_embeddings(model.rel_embeddings)
    dev_dataset.set_relation_embeddings(model.rel_embeddings)

    trainer = Trainer(model, train_dataset, dev_dataset, config)
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
