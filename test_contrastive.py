import time
import torch
import argparse
import representation
from gnns.loss import *
from gnns.gnn import Model
from gnns.contrastive_train_eval import train, evaluate
from util.stats import *
from util.save_load import *
from dataset.contrastive_dataset import get_loaders_from_args_gnn
from dataset.goose_domain_info import GOOSE_DOMAINS
from representation import REPRESENTATIONS_STR

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", choices=["ipc"] + GOOSE_DOMAINS)

    # model params
    parser.add_argument("-L", "--nlayers", type=int, default=4)
    parser.add_argument("-H", "--nhid", type=int, default=64)
    parser.add_argument(
        "--aggr",
        type=str,
        default="mean",
        choices=["sum", "mean", "max"],
        help="MPNN aggregation function.",
    )
    parser.add_argument(
        "--pool",
        type=str,
        default="sum",
        choices=["sum", "mean", "max"],
        help="Pooling function for readout. Always used sum in AAAI-24",
    )

    # optimisation params
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--reduction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)

    # data arguments
    parser.add_argument(
        "-r",
        "--rep",
        type=str,
        default="clg",
        choices=REPRESENTATIONS_STR,
        help="graph representation of planning tasks",
    )
    parser.add_argument(
        "-p",
        "--planner",
        type=str,
        default="fd",
        choices=["fd", "pwl"],
        help="for converting plans to states",
    )
    parser.add_argument(
        "--small-train",
        action="store_true",
        help="Small training set: useful for debugging.",
    )

    # save file
    parser.add_argument("--save-file", dest="save_file", type=str, default=None)

    # gpu device (if exists)
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)

    train_loader, val_loader = get_loaders_from_args_gnn(args)

    data, pos = train_loader.dataset[0]
    in_feat = data.x.shape[1]

    model_params = {
        "in_feat": in_feat,
        "out_feat": 1,
        "nlayers": 4,
        "n_edge_labels": 6,
        "nhid": 64,
        "aggr": "mean",
        "pool": "sum",
        "rep": "llg",
    }
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model(params=model_params).to(device)

    lr = 0.001
    reduction = 0.1
    patience = 10
    epochs = 10

    criterion = InfoNCELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", verbose=True, factor=reduction, patience=patience
    )
    for e in range(epochs):
        train_stats = train(model, device, train_loader, criterion, optimiser)
        print(train_stats)
