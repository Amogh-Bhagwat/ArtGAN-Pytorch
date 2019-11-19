import numpy as np
import torch
import torchvision
from torchvision import transforms

def get_dataset(root, batch_size, train=True):
	# Define transforms for cifar 10 dataset
	data_transform = transforms.Compose([
			# Resizing to 64, 64
			transforms.Resize((64, 64)),
			# converting to torch tensor
			transforms.ToTensor()
		])
	cifar_10_dataset = torchvision.datasets.CIFAR10(root=root, 
		train=train,
		transform=data_transform)
	dataloader = torch.utils.data.DataLoader(cifar_10_dataset, batch_size=batch_size, shuffle=True)

	return dataloader