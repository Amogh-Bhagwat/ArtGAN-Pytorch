import numpy as np
import torch
import sys
sys.path.insert(1, '../network')
from data_loader import get_dataset
from network import Generator, Discriminator, log_sum_exp 
from torch.nn.functional import one_hot
import torch.nn as nn

# Parameters
init_epoch = 0 
max_epoch = 100
store_image_epoch = 10
save_epoch = 10

lr = 0.0002
batch_size = 100
zdim = 100
n_classes = 23

# Dataloader
folder = "../data/wikiart/"
file_name = "../data/wikiart_csv/artist_train.csv"
train_loader = get_dataset(file_name, folder, batch_size)

# Network 
gen = Generator(in_channels=zdim+n_classes)
dis = Discriminator(classes=n_classes)

d_optimizer = torch.optim.RMSprop(dis.parameters(), lr=lr)
g_optimizer = torch.optim.RMSprop(gen.parameters(), lr=lr)


for index in range(init_epoch, max_epoch):
	for i, data in enumerate(train_loader):
		# Define input variables 
		x_n = data[0]
		y = torch.tensor(np.array(data[1], dtype=np.float32))
		batch_size = y.size()[0]
		z = torch.FloatTensor(batch_size, zdim).uniform_(-1, 1)
		# iny - one-hot encoded tensor for class labels
		iny = np.tile(np.eye(n_classes, dtype=np.float32), [int(batch_size / n_classes + 1), 1])[:batch_size, :]
		gen_inp = torch.cat((z, torch.tensor(iny)), axis=1)

		pred_n, recons_n = dis.forward(x_n)
		samples = gen.forward(gen_inp)
		pred_g, recons_g = dis.forward(samples)

		# Define generator and discriminator loss
		lreal = log_sum_exp(pred_n)
		lfake = log_sum_exp(pred_g)
		cost_pred_n = nn.CrossEntropyLoss(reduction='mean')(pred_n, y.to(torch.int64))
		cost_recons_n = torch.mean((recons_n - x_n.to(torch.float32)) ** 2) * 0.5
		cost_recons_g = torch.mean((recons_g - samples) ** 2) * 0.5 
		cost_dis_n = -torch.mean(lreal) + torch.mean(nn.Softplus()(lreal))
		cost_dis_gen_fake = torch.mean(nn.Softplus()(lfake))
		cost_dis = cost_dis_n + cost_dis_gen_fake + cost_pred_n + cost_recons_n

		cost_dis_g = -torch.mean(lfake) + torch.mean(nn.Softplus()(lfake))
		cost_pred_g = nn.CrossEntropyLoss(reduction='mean')(pred_g, torch.argmax(torch.tensor(iny), dim=1))   ### Problem in calculating this loss 
		cost_gen = cost_dis_g + cost_pred_g + cost_recons_g

		d_optimizer.zero_grad()
		g_optimizer.zero_grad()

		cost_dis.backward(retain_graph=True)

		cost_gen.backward()
		g_optimizer.step()

		print('Epoch:{}, G_loss={}, D_loss={}'.format(index, cost_gen.item(), cost_dis.item()))
