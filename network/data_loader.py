import numpy as np
import torch
import csv
import os
import pathlib
import PIL
from PIL import Image, ImageFile
# Code to load Wikiart Dataset 

def read_file(file_name, folder):
	images = []
	labels = []
	csv_file_path = file_name
	with open(csv_file_path) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			images.append(row[0])
			labels.append(row[1])
			line_count += 1 
		print('Processed {} lines'.format(line_count))

	return images, labels

class Dataset:
	def __init__(self, file_name, folder, transform=None):
		ImageFile.LOAD_TRUNCATED_IMAGES = True
		images, labels = read_file(file_name, folder)
		self.image_list = images
		self.labels = labels
		self.image_folder = folder

	def __getitem__(self, index):
		image_file = self.image_folder + self.image_list[index]
		image = Image.open(str(image_file))
		# resize the image to 64x64
		image = image.resize((64,64), PIL.Image.BILINEAR)
		image = np.array(image)
		# Convert image array to CHW format\
		image = np.transpose(image, (2, 0, 1))
		label = self.labels[index]
		return image, label

	def __len__(self):
		return len(self.labels)

def get_dataset(file_name, folder, batch_size):
	wikiart_dataset = Dataset(file_name, folder)
	dataloader = torch.utils.data.DataLoader(wikiart_dataset, batch_size=batch_size, shuffle=False)

	return dataloader
