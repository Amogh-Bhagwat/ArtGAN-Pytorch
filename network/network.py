import numpy as np 
import torch
import torch.nn as nn
import torch.distributions as tdist
"""
Network definitions for generator and discriminator
"""

class Generator(nn.Module):
	def __init__(self, in_channels):
		super().__init__()
		self.model = nn.Sequential(
			nn.Linear(in_channels, 512*4*4),
			nn.BatchNorm1d(512*4*4),
			nn.LeakyReLU(negative_slope=0.2),

			nn.UpsamplingNearest2d(scale_factor=8),

			nn.Conv2d(in_channels=512*4*4, out_channels=512, kernel_size=3, stride=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(negative_slope=0.2),

			nn.UpsamplingNearest2d(scale_factor=2),

			nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(negative_slope=0.2),

			nn.UpsamplingNearest2d(scale_factor=2),

			nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(negative_slope=0.2),

			nn.UpsamplingNearest2d(scale_factor=2),

			nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(negative_slope=0.2),

			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(negative_slope=0.2),
		)
	def forward(self, x):
		# x is the final concatenated vector of Normalised noise and lable information
		output = self.model(x)
		return output

class Discriminator(nn.Module):
	def __init__(self, classes):
		super().__init__()
		self.classes = classes
		self.model_cls = nn.Sequential(
			# 64
			nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=2),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Dropout2d(p=0.2),
			# 32
			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Dropout2d(p=0.2),
			# 16
			nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(negative_slope=0.2),
			# 8
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Dropout2d(p=0.2),
			# 8
			nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2),
			nn.BatchNorm2d(1024),
			nn.LeakyReLU(negative_slope=0.2)
			# 4 
			# nn.Linear(1024, classes)
			# nn.LeakyReLU(negative_slope=0.2)
		)

		self.model_decoder = nn.Sequential(
			nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(negative_slope=0.2),

			nn.UpsamplingNearest2d(scale_factor=2),
			# 8
			nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(negative_slope=0.2),

			nn.UpsamplingNearest2d(scale_factor=2),
			# 16
			nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(negative_slope=0.2),

			nn.UpsamplingNearest2d(scale_factor=2),
			# 32
			nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(negative_slope=0.2),

			nn.UpsamplingNearest2d(scale_factor=2),
			# 64
			nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(negative_slope=0.2),


			nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1),
			# Change to tanh actuvation function
			nn.LeakyReLU(negative_slope=0.2)
		)
	def forward(self, x):
		# Denoiser
		x = x + tdist.Normal(torch.tensor([0.0]), torch.tensor([0.05])).sample(x.shape)
		# Pass through network
		output = self.model_cls(x)
		# Get class prob 
		class_prob = nn.Sequential(nn.Linear(1024, self.classes),
			nn.LeakyReLU(negative_slope=0.2)
			)(output)
		recons = self.model_decoder(output)

		return class_prob, recons

def log_sum_exp(inp, axis=1):
	m = np.max(inp, axis=axis, keep_dims=True)
	return m + torch.log(np.sum(torch.exp(inp - m), axis=axis))
	