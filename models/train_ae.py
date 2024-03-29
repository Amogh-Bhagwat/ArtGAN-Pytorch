import numpy as np
import torch
import sys
sys.path.insert(1, '../network')
from data_loader_cifar10 import get_dataset
from network_ae import Generator, Discriminator, log_sum_exp 
from torch.nn.functional import one_hot
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch.nn.functional as F

# Parameters
init_epoch = 0 
max_epoch = 10
store_image_epoch = 10
save_epoch = 1

lr_init = 0.002
batch_size = 128
zdim = 100
n_classes = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataloader
# folder = "../data/wikiart/"
# file_name = "../data/wikiart_csv/artist_train.csv"
# train_loader = get_dataset(file_name, folder, batch_size)

root = "../data/"
train_loader = get_dataset(root, batch_size)
# Network 
gen = Generator(in_channels=zdim+n_classes).to(device)
dis = Discriminator(classes=n_classes + 1, device=device).to(device) # Additional fake class

d_optimizer = torch.optim.RMSprop(dis.parameters(), lr=lr_init)
g_optimizer = torch.optim.RMSprop(gen.parameters(), lr=lr_init)

gen_loss = []
dis_loss = []

for index in range(init_epoch, max_epoch):
	for i, data in enumerate(train_loader):
		if index >= 8:
			for g in g_optimizer.param_groups:
				g['lr'] = lr_init / 10
			for g in d_optimizer.param_groups:
				g['lr'] = lr_init / 10
		# Define input variables 
		x_n = torch.tensor(np.array(data[0], dtype=np.float32), device=device)
		y = torch.tensor(np.array(data[1], dtype=np.float32), device=device)
		y_one_hot = one_hot(y.to(torch.int64), n_classes + 1) # Additional fake class 
		batch_size = y.size()[0] 
		z = torch.FloatTensor(batch_size, zdim).uniform_(-1, 1).to(device)
		# iny - one-hot encoded tensor for class labels
		iny = torch.tensor(np.tile(np.eye(n_classes, dtype=np.float32), [int(batch_size / n_classes + 1), 1])[:batch_size, :], device=device)
		gen_inp = torch.cat((z, iny), dim=1)
		y_fake = torch.tensor(np.full((1, batch_size), n_classes), device=device)
		y_fake_one_hot = one_hot(y_fake.to(torch.int64), n_classes + 1)

		pred_n, recons_n = dis.forward(x_n)
		samples = gen.forward(gen_inp)
		# print(samples)
		pred_g, recons_g = dis.forward(samples)

		# Define generator and discriminator loss
		# lreal = log_sum_exp(pred_n)
		# lfake = log_sum_exp(pred_g)
		cost_pred_n = F.binary_cross_entropy(pred_n, y_one_hot.to(torch.float32)) # cost for real image prediction
		cost_recons_n = torch.mean((recons_n - x_n) ** 2)			      # reconstruction losses 
		cost_recons_g = torch.mean((recons_g - samples) ** 2)  
		cost_pred_f = F.binary_cross_entropy(pred_g, y_fake_one_hot.to(torch.float32))
		# cost_dis_n = -torch.mean(lreal) + torch.mean(nn.Softplus()(lreal))
		# cost_dis_gen_fake = torch.mean(nn.Softplus()(lfake))
		cost_dis = cost_pred_f + cost_pred_n + cost_recons_n

		# cost_dis_g = -torch.mean(lfake) + torch.mean(nn.Softplus()(lfake))
		zeros = torch.tensor(np.full((batch_size, 1), 0), device=device)
		y_fake_loss = torch.cat((iny, zeros.to(torch.float32)), dim=1)
		cost_pred_g = F.binary_cross_entropy(pred_g, y_fake_loss)   # adversarial loss for generator
		cost_gen = cost_pred_g + cost_recons_g

		d_optimizer.zero_grad()

		cost_dis.backward(retain_graph=True)
		d_optimizer.step()

		g_optimizer.zero_grad()

		cost_gen.backward()
		g_optimizer.step()

		print('Epoch:{}, Iter:{}, G_loss={}, D_loss={}'.format(index, i, cost_gen.item(), cost_dis.item()))
		gen_loss.append(cost_gen.item())
		dis_loss.append(cost_dis.item())
	# get images to test progress of generator and save generator 
	if index % save_epoch == 0 or index == max_epoch -1:
		path = "../saved_models/" + str(index)
		torch.save(gen.state_dict(), path)

		with torch.no_grad():
			samples = samples.detach().cpu()
		img_list = vutils.make_grid(samples, nrow=10, padding=2).numpy()
		plt.imshow(((np.transpose(img_list, (1,2,0)) + 1.) * 127.5).astype(int), interpolation="nearest")
		path = "../gen_images/artgan_ae" + str(index) + ".png"
		plt.savefig(path)

# Save loss data for plots
with open('gen_loss.txt', 'w') as f:
    for item in gen_loss:
        f.write("%s\n" % item)
with open('dis_loss.txt', 'w') as f:
    for item in dis_loss:
        f.write("%s\n" % item)