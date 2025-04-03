import datetime

import datasets
import models
import torch

def train_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs+1):
        loss_train = 0.0
        for batch, predicts in train_loader:
            outputs  = model(batch)
            loss = loss_fn(outputs, predicts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        if epoch ==1 or epoch % 10 == 0:
            print('{} Epoch {}, training loss {}'.format(datetime.datetime.now(), epoch, loss_train/len(train_loader)))

#
samples = datasets.SpyDailyDataset()
train_loader = torch.utils.data.DataLoader(samples, batch_size=64, shuffle=True)
model = models.Conv1DNet(features=len(samples.feature_labels))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_fn = torch.nn.MSELoss()
#
train_loop(n_epochs=100, optimizer=optimizer, model=model, loss_fn=loss_fn, train_loader=train_loader)