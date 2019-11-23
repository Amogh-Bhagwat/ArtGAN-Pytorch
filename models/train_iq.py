import torch
import numpy as np
import sys
sys.path.insert(1, '../network')
from data_loader_cifar10 import get_dataset
from network_iq import Generator, Discriminator
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = "../data/"

z_dim = 100
n_classes = 10
init_epoch = 0
max_epoch = 10
batch_size = 128
save_epoch = 1
in_channels = z_dim + n_classes
out_channels = n_classes + 1 # for fake class 
lr_init = 0.001

train_loader = get_dataset(root, batch_size)

gen = Generator(in_channels).to(device)
dis = Discriminator(out_channels).to(device)

g_optimizer = torch.optim.RMSprop(gen.parameters(), lr=lr_init, alpha=0.9)
d_optimizer = torch.optim.RMSprop(dis.parameters(), lr=lr_init, alpha=0.9)

gen_loss = []
dis_loss = []

for index in range(init_epoch, max_epoch):
	for i, data in enumerate(train_loader):
		# lr reduced by factor of 10 after 8 epoch
		if index >= 8:
			for g in g_optimizer.param_groups:
				g['lr'] = lr_init / 10
			for g in d_optimizer.param_groups:
				g['lr'] = lr_init / 10
		# Define input variables
		x_n = torch.tensor(np.array(data[0], dtype=np.float32), device=device)
		y = torch.tensor(np.array(data[1], dtype=np.float32), device=device)
		y_one_hot = F.one_hot(y.to(torch.int64), n_classes + 1)
		batch_size = y.size()[0]
		y_fake = torch.tensor(np.full((1, batch_size), n_classes), device=device)
		y_fake_one_hot = F.one_hot(y_fake.to(torch.int64), n_classes + 1)
		z = torch.FloatTensor(batch_size, z_dim).uniform_(-1, 1).to(device)
		iny = torch.tensor(np.tile(np.eye(n_classes, dtype=np.float32), [int(batch_size / n_classes + 1), 1])[:batch_size, :], device=device)		
		gen_inp = torch.cat((z, iny), dim=1)

		pred_real = dis.forward(x_n, only_encoder=False)
		fake_imgs, fake_imgs_128 = gen.forward(gen_inp, only_decoder=False)
		pred_fake = dis.forward(fake_imgs, only_encoder=False)

		# Define discriminator loss and update discriminator
		loss_real = F.binary_cross_entropy(pred_real, y_one_hot.to(torch.float32))
		loss_fake = F.binary_cross_entropy(pred_fake, y_fake_one_hot.to(torch.float32))
		loss_dis = loss_fake + loss_real
		d_optimizer.zero_grad()
		loss_dis.backward(retain_graph=True) 
		d_optimizer.step()

		# Define adversarial loss
		zeros = torch.tensor(np.full((batch_size, 1), 0), device=device)
		y_fake_loss = torch.cat((iny, zeros.to(torch.float32)), dim=1)
		loss_gen_fake = F.binary_cross_entropy(pred_fake, y_fake_loss.to(torch.float32))
		# Reconstruction and l2 loss
		recons = gen.forward(dis.forward(x_n, only_encoder=True), only_decoder=True)
		loss_l2 = torch.mean((recons - x_n) ** 2)
		loss_gen = loss_gen_fake + loss_l2
		g_optimizer.zero_grad()
		loss_gen.backward()
		g_optimizer.step()

		print('Epoch:{}, Iter:{}, G_loss={}, D_loss={}'.format(index, i, loss_gen.item(), loss_dis.item()))

	if index % save_epoch == 0 or index == max_epoch -1:
		path = "../saved_models/artgan_iq" + str(index)
		torch.save(gen.state_dict(), path)

		# Save images from generator
		with torch.no_grad():
			samples = fake_imgs_128.detach().cpu()
		img_list = vutils.make_grid(samples, nrow=10, padding=2).numpy()
		plt.imshow(((np.transpose(img_list, (1,2,0)) + 1.) * 127.5).astype(int), interpolation="nearest")
		path = "../gen_images/artgan_iq" + str(index) + ".png"
		plt.savefig(path)


with open('gen_loss_artgan_iq.txt', 'w') as f:
    for item in loss_gen:
        f.write("%s\n" % item)
with open('dis_loss_artgan_iq.txt', 'w') as f:
    for item in loss_dis:
        f.write("%s\n" % item)