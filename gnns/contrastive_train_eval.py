import torch

""" Train and evaluation methods in training pipeline. """


def train(model, device, train_loader, criterion, optimiser):
    model.train()
    train_loss = 0

    for data in train_loader:
        data, pos = data
        data = data.to(device)
        pos = pos.to(device)

        optimiser.zero_grad(set_to_none=True)

        embeddings = model.forward_to_embeddings(data)
        with torch.no_grad():
            pos_embeddings = model.forward_to_embeddings(pos)

        loss = criterion.forward(embeddings, pos_embeddings)

        loss.backward()
        optimiser.step()
        train_loss += loss.detach().cpu().item()

    stats = {
        "loss": train_loss / len(train_loader),
    }
    return stats


@torch.no_grad()
def evaluate(model, device, val_loader, criterion, return_true_preds=False):
    model.eval()
    val_loss = 0

    if return_true_preds:
        y_true = torch.tensor([])
        y_pred = torch.tensor([])

    for data in val_loader:
        data = data.to(device)
        h_true = data.y.float().to(device)

        h_pred = model.forward(data)
        loss = criterion.forward(h_pred, h_true)

        # embeddings = model.forward_to_embeddings(data)
        # h_pred = model.forward_from_embeddings(embeddings)
        # loss = criterion.forward(embeddings, h_pred, h_true)
        
        val_loss += loss.detach().cpu().item()

        if return_true_preds:
            y_pred = torch.cat((y_pred, h_pred.detach().cpu()))
            y_true = torch.cat((y_true, h_true.detach().cpu()))

    stats = {
        "loss": val_loss / len(val_loader),
    }
    if return_true_preds:
        stats["y_true"] = y_true
        stats["y_pred"] = y_pred

    return stats
