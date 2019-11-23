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
		# self.model_fc = nn.Sequential(
		# 	nn.Linear(in_channels, 512*4*4),
		# 	nn.BatchNorm1d(512*4*4),
		# 	nn.LeakyReLU(negative_slope=0.2)
		# )
		self.in_channels = in_channels
		self.model_gen = nn.Sequential(	
			#nn.UpsamplingNearest2d(scale_factor=2),
			nn.ConvTranspose2d(self.in_channels, 1024, 4, 1, 0),
			nn.BatchNorm2d(1024),
			nn.ReLU(),

			nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(),

			# nn.UpsamplingNearest2d(scale_factor=2),

			nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),

			# nn.UpsamplingNearest2d(scale_factor=2),

			nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),

			# nn.UpsamplingNearest2d(scale_factor=2),

			nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),

			nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
			# nn.BatchNorm2d(3),
			# nn.LeakyReLU(negative_slope=0.2)  # change the activation function to tanh
			nn.Sigmoid() 
		)
	def forward(self, x):
		# x is the final concatenated vector of Normalised noise and lable information
		# output_fc = self.model_fc(x)
		# reshape output to 512 x 4 x 4 
		# output = output_fc.view(-1, 512, 4, 4)
		batch_size = x.shape[0]
		n_channels = x.shape[1]
		x = x.view(batch_size, n_channels, 1, 1)
		output = self.model_gen(x)
		return output

class Discriminator(nn.Module):
	def __init__(self, classes, device):
		super().__init__()
		self.classes = classes
		self.device = device
		self.model_cls = nn.Sequential(
			# 64
			nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1),
			nn.LeakyReLU(negative_slope=0.2),
			#nn.Dropout2d(p=0.2),
			# 32
			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(negative_slope=0.2),
			#nn.Dropout2d(p=0.2),
			# 16
			nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(negative_slope=0.2),
			# 8
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(negative_slope=0.2),
			#nn.Dropout2d(p=0.2),
			# 8
			nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(1024),
			nn.LeakyReLU(negative_slope=0.2)
			# 4 
			# nn.Linear(1024, classes)
			# nn.LeakyReLU(negative_slope=0.2)
		)

		self.model_decoder = nn.Sequential(
			nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(negative_slope=0.2),

			#nn.UpsamplingNearest2d(scale_factor=2),
			# 8
			nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(negative_slope=0.2),

			# nn.UpsamplingNearest2d(scale_factor=2),
			# 16
			nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(negative_slope=0.2),

			# nn.UpsamplingNearest2d(scale_factor=2),
			# 32
			nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(negative_slope=0.2),

			# nn.UpsamplingNearest2d(scale_factor=2),
			# 64
			nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(negative_slope=0.2),


			nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
			# Change to tanh actuvation function
			# nn.LeakyReLU(negative_slope=0.2)
			nn.Sigmoid()
		)
	def forward(self, x):
		# Denoiser
		# print(x.shape)
		# x = x.to(torch.float32) + tdist.normal.Normal(torch.tensor([0.0]), torch.tensor([0.05])).rsample(x.shape).squeeze(4).to(self.device)
		output = self.model_cls(x)
		# Get class prob
		class_prob_net = nn.Sequential(
				nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=4, stride=2, padding=1),
				nn.BatchNorm2d(1024),
				nn.LeakyReLU(0.2),

				nn.Conv2d(in_channels=1024, out_channels=self.classes, kernel_size=4, stride=2, padding=1),
				# nn.Linear(1024*4*4, self.classes),
				# nn.LeakyReLU(negative_slope=0.2)
				nn.Sigmoid()
			)
		class_prob = class_prob_net.to(self.device)(output)
		recons = self.model_decoder(output)
		return class_prob, recons

def log_sum_exp(inp, axis=1):
	m = torch.max(inp, dim=axis, keepdim=True)
	return m.values + torch.log(torch.sum(torch.exp(inp - m.values), dim=axis))
	