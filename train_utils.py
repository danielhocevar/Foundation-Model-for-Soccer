import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def accuracy(preds, labels):
    return sum(preds[:, -1].argmax(dim=1) == labels)

def train(model, device, train_loader, val_loader, len_train, len_val, n_epochs=20, lr=0.0002, save_name="weights"):
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(n_epochs):
        model.train()
        i = 0
        train_loss = 0
        train_acc = 0
        for x_bat, y_bat in tqdm(iter(train_loader)):
            x_bat = x_bat.to(device)
            x_bat.permute(1, 0)
            y_bat = y_bat.to(device)
            optimizer.zero_grad()
            y_pred = model(x_bat)
            loss = criterion(y_pred[:, -1, :], y_bat)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            i += 1
            train_acc += accuracy(y_pred, y_bat)

            # print(f'Epoch {epoch}, iter {i}, loss: {loss.item()}')

        train_acc = train_acc / len_train #len(X_train[:-1])
        train_loss = train_loss / len(train_loader)
        model.eval()
        val_loss = 0
        val_acc = 0
        for x_val, y_val in tqdm(iter(val_loader)):
            x_val = x_val.to(device)
            y_val = y_val.to(device)

            y_pred = model(x_val)
            loss = criterion(y_pred[:, -1, :], y_val)
            val_loss += loss.item()
            val_acc += accuracy(y_pred, y_val)

        val_acc = val_acc / len_val #len(X_val[:-1])
        val_loss = val_loss / len(val_loader)  
        print(f'Epoch {epoch}, iter {i}, train_loss: {train_loss}, train_acc: {train_acc}, val_loss: {val_loss}, val_acc: {val_acc}')
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        torch.save(model.state_dict(), f'{save_name}_epoch{epoch}.pt')

    return train_losses, train_accs, val_losses, val_accs