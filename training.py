import hydra
from hydra.utils import instantiate
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data, HeteroData
from dataset import MoleculeDataset
from utils import random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math


def train_epoch(model, optimizer, loader, criterion):
    model.train()
    device = next(model.parameters()).device
    losses = []
    for x in loader:
        for key in x:
            if isinstance(x[key], torch.Tensor) or isinstance(x[key], Data) or isinstance(x[key], HeteroData):
                x[key] = x[key].to(device)

        optimizer.zero_grad()
        out = model(**x)
        loss = criterion(out, x['U_0'])
        loss.backward()
        optimizer.step()
        losses += [loss.item()]
    return losses

def val_epoch(model, loader, criterion):
    model.eval()
    device = next(model.parameters()).device
    sum_losses = 0.0
    with torch.no_grad():
        for x in loader:
            for key in x:
                if isinstance(x[key], torch.Tensor) or isinstance(x[key], Data) or isinstance(x[key], HeteroData):
                    x[key] = x[key].to(device)

            out = model(**x)
            loss = criterion(out, x['U_0'])
            sum_losses += loss.item() * x['num']
    return [sum_losses / len(loader.dataset)]

def train(model, optimizer, train_loader, val_loader, criterion, num_epochs, save_path, lr_scheduler=None, early_stop=None):
    train_losses, val_losses = [], []
    best_loss = None
    for _ in tqdm(range(num_epochs)):
        train_losses += train_epoch(model, optimizer, train_loader, criterion)
        val_losses += val_epoch(model, val_loader, criterion)
        if best_loss is None or val_losses[-1] < best_loss:
            best_loss = val_losses[-1]
            state = {
                "model state_dict": model.state_dict(),
                "optimizer state_dict": optimizer.state_dict(),
                "epoch": len(val_losses),
                "train_losses": train_losses,
                "val_losses": val_losses
            }
            if lr_scheduler is not None:
                state["lr_scheduler state_dict"] = lr_scheduler.state_dict()
            torch.save(state, save_path)
        if early_stop is not None:
            if len(val_losses) >= early_stop + 1 and val_losses[-early_stop - 1] <= val_losses[-1]:
                break
        if lr_scheduler is not None:
            lr_scheduler.step()
    return train_losses, val_losses

@hydra.main(version_base=None, config_path="conf", config_name="homo_gnn")
def train_and_eval(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = instantiate(cfg.dataset)
    random_state = cfg.random_state if "random_state" in cfg else None

    # fix random seeds for reproducibility
    torch.manual_seed(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # np.random.seed(random_state)
    # random.seed(random_state)

    train_ds, test_ds, val_ds = random_split(ds, cfg.split_ratios, random_state=random_state)
    train_ds.train()
    test_ds.eval()
    val_ds.eval()
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers if "num_workers" in cfg else 1
    collate_fn = instantiate(cfg.collate_fn, _partial_=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)
    
    model = instantiate(cfg.model).to(device)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    lr_scheduler = instantiate(cfg.lr_scheduler, optimizer=optimizer) if "lr_scheduler" in cfg else None
    criterion = instantiate(cfg.criterion)
    
    early_stop = cfg.early_stop if "early_stop" in cfg else None
    num_epochs = cfg.num_epochs
    save_path = cfg.save_path
    train_losses, val_losses = train(model, optimizer, train_loader, val_loader, criterion, num_epochs, save_path, lr_scheduler, early_stop)
    model.load_state_dict(torch.load(save_path)["model state_dict"])
    print(f"{cfg.name} loss: ", end='')
    print(val_epoch(model, test_loader, criterion)[0])
    plt.plot(train_losses, label='train')
    plt.plot(math.ceil(len(train_loader.dataset) / batch_size) * torch.arange(len(val_losses)), val_losses, label='val')
    plt.legend()
    plt.yscale('log')
    image_path = cfg.image_path
    plt.savefig(image_path, bbox_inches='tight')
