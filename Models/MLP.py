import torch as pt
from torch import nn, optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class MLP(nn.Module):
  def __init__(self, input_dim: int, output_dim: int):
    super().__init__()
    # self.layers = nn.Sequential(
    #   nn.Linear(input_dim, 5),
    #   nn.ReLU(),
    #   nn.Linear(5, output_dim),
    #   nn.Sigmoid()
    # )
    
    self.layers = nn.Sequential(
      nn.Linear(input_dim, 10),
      nn.ReLU(),
      nn.Linear(10, 5),
      nn.ReLU(),
      nn.Linear(5, output_dim),
      nn.Sigmoid()
    )

  def forward(self, x):
    return self.layers(x)
    
# def training(model: nn.Module, dataset: Dataset, max_epochs: int, loss_list: list | None = None):
def training(model: nn.Module, dataset: Dataset, max_epochs: int,testDataset: Dataset, loss_list: list = None,val_loss_list = None, lr = 0.01,printLoss = False):
    #data precessor
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(testDataset, batch_size=64, shuffle=False)
    # print("lr: ", lr)
    loss: pt.Tensor
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr,weight_decay=0.001)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=10,min_lr=1e-06)
    # model.train()
    epoch_loss = []
    # print("testData: ",testData)

    for _ in range(max_epochs):
      model.train()
      running_loss = 0.0
      #把全部資料分成許多batch做訓練
      for X_batch,Y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
      # optimizer.zero_grad()
        loss = criterion(outputs, Y_batch)
        running_loss += loss.item() * X_batch.size(0)
        loss.backward()
        optimizer.step()
      train_loss = running_loss / len(train_loader.dataset)
      epoch_loss.append(train_loss)
      
      model.eval()
      val_loss = 0.0
      with pt.no_grad():
        for X_batch,Y_batch in test_loader:
                # outputs = model(testDataset.x)
                # validloss = criterion(outputs,testDataset.y)
                outputs = model(X_batch).squeeze()
                validLoss = criterion(outputs,Y_batch)
                val_loss += validLoss.item() * X_batch.size(0)

      val_loss = val_loss / len(test_loader.dataset)
      # optimizer.zero_grad()
      scheduler.step(val_loss)
      # if loss_list is not None:
      #     loss_list.append(loss.item())
      # print("printLoss: ",printLoss)
      # if ((_+1) % 3 == 0) & printLoss == True:
      #   print("aaa")
      if ( ( (_+1) % 15 == 0) | (_ == 0) ) & printLoss == True:
        print(f"Epoch {_+1}/{max_epochs}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    if loss_list is not None:
        loss_list.append(train_loss)
    if val_loss_list is not None:
        val_loss_list.append(val_loss)
    if loss_list is not None:
        plt.figure()
        plt.plot(epoch_loss)
        plt.xlabel('Round')
        plt.ylabel('loss')
        plt.title('loss')
        plt.savefig('MLP_loss.png')
        plt.close()
    # model.eval()
    