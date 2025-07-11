import numpy as np
import torch
import torch.nn.functional as F

from rfstudio.nn.utils.curve_mapping import CurveMapping

if __name__ == '__main__':
    mapping = CurveMapping(feature_dim=1)
    mapping.__setup__()
    optimizer = torch.optim.Adam(mapping.parameters(), lr=0.1)
    max_exp = 5
    for i in range(100):
        optimizer.zero_grad()
        train_data = torch.rand((1000, 1))
        train_label = ((train_data * max_exp).exp() - 1) / (np.exp(max_exp) - 1)
        train_pred = mapping(train_data)
        loss = F.l1_loss(train_label, train_pred)
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.01
