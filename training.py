import time
import math
from torch.utils.data import random_split, DataLoader

import datasets
import models
import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda") 
BATCH_SIZE=1024

samples = datasets.SpyDailyDataset(device=device) 
train_size = int(0.7 * len(samples))
val_size = int(0.15 * len(samples))
test_size = len(samples) - train_size - val_size

train_set, val_set, test_set = random_split(
    samples, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,  shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

model = models.Conv1DNet(
    in_chans=len(samples.feature_labels), 
    out_chans=len(samples.prediction_labels),
    n_lookbehind=samples.n_lookbehind).to(device=device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

train_loss = torch.nn.MSELoss()
validate_loss = torch.nn.L1Loss()


def validate(model):
    for name, loader in [("training", train_loader), ("validation", val_loader)]:
        with torch.no_grad():
            loss_val = 0.0
            for batch, predicts in val_loader:
                outputs = model(batch)
                loss = validate_loss(torch.exp(outputs), torch.exp(predicts))
                loss_val += loss.item()
            loss_val /= len(loader)
        print(f'{name} loss: {loss_val}')


def train_eval_loop_l2reg(n_epochs, optimizer, model, train_loader):
    start_time = time.time()
    for epoch in range(1, n_epochs+1):
        loss_train = 0.0
        for batch, predicts in train_loader:
            outputs  = model(batch)
            loss = train_loss(outputs, predicts)
            # add l2 regularization            
            l2_lambda = 0.01
            l2_norm = sum(p.pow(2).sum() for p in model.parameters())
            loss += l2_lambda * l2_norm
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        if epoch ==1 or epoch % 10 == 0:
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"Epoch {epoch}/{n_epochs}, time eplase per epoch {elapsed/epoch:.2f}s")
            validate(model)   
    

#
train_eval_loop_l2reg(n_epochs=100, optimizer=optimizer, model=model, train_loader=train_loader)
# save model
torch.save(model.state_dict(), "./data/conv1d.pth")