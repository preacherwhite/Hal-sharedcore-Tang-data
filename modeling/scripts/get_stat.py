import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from modeling.models.cnns.bethge import BethgeModel
from modeling.train_utils import array_to_dataloader
nb_validation_samples = 1000
train_y = np.load('Rsp.npy')
val_y = np.load('valRsp.npy')
batch_size = 2048
num_neurons = 299
train_x = np.load('train_x.npy')
val_x = np.load('val_x.npy')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

channels = 300
num_layers = 12
input_size = 50
output_size = 299
first_k = 5
later_k = 3
pool_size = 2
factorized = True
num_maps = 1

net = BethgeModel(channels=channels, num_layers=num_layers, input_size=input_size,
            output_size=output_size, first_k=first_k, later_k=later_k,
            input_channels=1, pool_size=pool_size, factorized=True,
            num_maps=num_maps).cuda()


net.to(device)
train_loader = array_to_dataloader(train_x, train_y, batch_size=1024)
val_loader = array_to_dataloader(val_x, val_y, batch_size=1024)

net.load_state_dict(torch.load('shared_core_300_12_model'))

with torch.no_grad():
    net.eval()
    prediction = []
    actual = []
    for batch_num, (x, y) in enumerate(tqdm(val_loader)):
        x, y = x.to(device), y.to(device)
        outputs = net(x).cpu().numpy()
        prediction.extend(outputs)
        actual.extend(y.cpu().numpy())

    prediction = np.stack(prediction)
    actual = np.stack(actual)
    R = np.zeros(299)
    VE = np.zeros(299)
    for neuron in range(num_neurons):
        pred1 = prediction[:, neuron]
        val_y = actual[:, neuron]
        y_arg = np.argsort(val_y)

        u2 = np.zeros((2, nb_validation_samples))
        u2[0, :] = np.reshape(pred1, (nb_validation_samples))
        u2[1, :] = np.reshape(val_y, (nb_validation_samples))

        c2 = np.corrcoef(u2)
        R[neuron] = c2[0, 1]

        VE[neuron] = 1 - np.var(pred1 - val_y) / np.var(val_y)

        plt.plot(pred1[y_arg],label='pred')
        plt.plot(np.sort(val_y),label='actual')
        plt.title(R[neuron])
        plt.legend()
        plt.savefig('pics1/'+str(neuron)+'tuning')
        plt.show()


    print(R)
    print(np.mean(R))
    print(np.mean(VE))



