import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
	def __init__(self, in_channels):
		super().__init__()
		# Input is noise vector concatenated with lable information
		self.in_channels = in_channels
		self.model_znet = nn.Sequential(
			nn.ConvTranspose2d(self.in_channels, 1024, 4, 1, 0),
			nn.BatchNorm2d(1024),
			nn.ReLU(negative_slope=0.2),

			nn.ConvTranspose2d(1024, 512, 4, 2, 1),
			nn.BatchNorm2d(512),
			nn.ReLU(negative_slope=0.2)			
		)
		self.model_decoder = nn.Sequential(
			nn.ConvTranspose2d(512, 256, 4, 2, 1),
			nn.BatchNorm2d(256),
			nn.ReLU(negative_slope=0.2),

			nn.ConvTranspose2d(256, 128, 4, 2, 1),
			nn.BatchNorm2d(128),
			nn.ReLU(negative_slope=0.2),

			nn.ConvTranspose2d(128, 128, 3, 1, 1),
			nn.BatchNorm2d(128),
			nn.ReLU(negative_slope=0.2),

			nn.ConvTranspose2d(128, 3, 4, 2, 1),
			nn.Sigmoid()
		)
		# output is 64 x 64 images
	def forward(x, only_decoder=False):
		if only_decoder:
			return self.model_decoder(x)
		else:
			out = self.model_znet(x)
			out = self.model_decoder(out)
			return out

class Discriminator(nn.Module):
	def __init__(self, out_channels):
		super().__init__()
		# Input is 3 x 64 x 64 images
		self.out_channels = out_channels
		self.model_encoder = nn.Sequential(
			nn.Conv2d(3, 128, 4, 2, 1),
			nn.LeakyReLU(0.2)

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

			nn.Flatten(),

			nn.Linear(4 * 4 * 1024, out_channels)
		)
	def forward(x, only_encoder=False):
		if only_encoder:
			return self.model_encoder(x)
		else:
			out = self.model_encoder(x)
			out = self.model_cls(out)
			return out
			