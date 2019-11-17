import numpy as np
import torch
import csv
import cv2
import os
import pathlib
from PIL import Image

# Code to load Wikiart Dataset 

def read_file(file_name, folder):
	images = []
	labels = []
	file_path = folder + file_name
	with open(file_path) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			images.append(row[0])
			labels.append(row[1])
			line_count += 1 
		print(f'Processed {line_count} lines')

	return images, labels

class Dataset:
	def __init__(self, file_name, folder, transform=None):
		images, labels = read_file(file_name, folder)
		self.image_list = images
		self.labels = labels
		self.image_folder = IMAGES_FOLDER

	def __getitem__(self, index):
		image_file = self.image_folder + self.image_list[index]
		image = cv2.imread(str(image_file))
		# resize the image to 64x64
		image = cv2.resize(image, (64,64))
		label = self.labels[index]
		return image, label

	def __len__(self):
		return len(self.labels)

def get_dataset(file_name, folder, batch_size):
	wikiart_dataset = Dataset(file_name, folder)
	dataloader = DataLoader(wikiart_dataset, batch_size=batch_size, shuffle=False)

	return dataloader
