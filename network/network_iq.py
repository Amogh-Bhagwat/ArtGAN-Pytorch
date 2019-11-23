import torch
import torch.nn as nn
import numpy as np


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class Generator(nn.Module):
	def __init__(self, in_channels):
		super().__init__()
		# Input is noise vector concatenated with lable information
		self.in_channels = in_channels
		self.model_znet = nn.Sequential(
			nn.ConvTranspose2d(self.in_channels, 1024, 4, 1, 0),
			nn.BatchNorm2d(1024),
			nn.ReLU(),

			nn.ConvTranspose2d(1024, 512, 4, 2, 1),
			nn.BatchNorm2d(512),
			nn.ReLU()			
		)
		self.model_decoder = nn.Sequential(
			nn.ConvTranspose2d(512, 256, 4, 2, 1),
			nn.BatchNorm2d(256),
			nn.ReLU(),

			nn.ConvTranspose2d(256, 128, 4, 2, 1),
			nn.BatchNorm2d(128),
			nn.ReLU(),

			nn.ConvTranspose2d(128, 64, 4, 2, 1),
			nn.BatchNorm2d(64),
			nn.ReLU(),

			nn.ConvTranspose2d(64, 64, 3, 1, 1),
			nn.BatchNorm2d(64),
			nn.ReLU(),

			nn.ConvTranspose2d(64, 3, 4, 2, 1),
			nn.Sigmoid()
		)
		# output is 128 x 128 images
	def forward(self, x, only_decoder=False):
		if only_decoder:
			out_128 = self.model_decoder(x)
			out = nn.AvgPool2d(2,2)(out_128)
			return out
		else:
			# reshape x to batch_size x num_channels x 1 x 1
			batch_size = x.shape[0]
			num_channels = x.shape[1]
			x = x.view(batch_size, num_channels, 1, 1)
			out_128 = self.model_znet(x)
			out_128 = self.model_decoder(out_128)
			out = nn.AvgPool2d(2,2)(out_128)
			return out, out_128

class Discriminator(nn.Module):
	def __init__(self, out_channels):
		super().__init__()
		# Input is 3 x 64 x 64 images
		self.out_channels = out_channels
		self.model_encoder = nn.Sequential(
			nn.Conv2d(3, 128, 4, 2, 1),
			nn.LeakyReLU(0.2),

			nn.Conv2d(128, 128, 3, 1, 1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2),

			nn.Conv2d(128, 256, 4, 2, 1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2),

			nn.Conv2d(256, 512, 4, 2, 1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2)
		)
		self.model_cls = nn.Sequential(
			nn.Conv2d(512, 1024, 4, 2, 1),
			nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.2),

			Flatten(),

			nn.Linear(4 * 4 * 1024, out_channels),
			nn.Sigmoid()
		)
	def forward(self, x, only_encoder=False):
		if only_encoder:
			return self.model_encoder(x)
		else:
			out = self.model_encoder(x)
			out = self.model_cls(out)
			return out
			