import os
import sys
import socket
import numpy as np

import torch
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torchvision import datasets
from torch.autograd import Variable
import torchvision.transforms as transforms

import time

from torch import nn
from torch import optim

import torch.nn.functional as F




################################################################################
##########          REPLICA
################################################################################

""" Definition of a Replica
class Replica ():

	# PRIVATE: constructor
	def __init__ (self, part_file="part.dist")

	# Get speeds
	def getSpeeds (self)

	# Setup dataset/dataloader for the replica
	def setData (self, loader)

	# Set model to train
	def setModel (self, model)

	# PRIVATE: Update gradients in each iteration of training
	def __update_gradients (self)

	# PRIVATE: write training performance to files
	def __benchmarkToFile (self, performance)

	# Run benchmarking on training
	def benchmark (self)

	# Run training (no benchmarking, except accuracy)
	def run (self)

	# Return performance data
	def getPerformance (self)
"""
def readtype():

	f = sys.argv[3]
	print(sys.argv[3], "--------------")
	f=open(f,"r")
	lines=f.readlines()
	types=[]
	i = 0
	for x in lines:
		if(i < len(lines) and i != 0):
			types.append(x.split(' ')[3])
		i = i + 1
	f.close()
	print(types)
	return types


class Replica ():

	def __init__ (self, part_file="part.dist", open=True,cluster = None):

		self.size = 1
		self.rank = 0
		self.rank_gpu = None
		self.hostname = socket.gethostname()

		self.performance = {
		'Time':      0.0,
		'EpochTime': [],
		'CompTime':  [],
		'CommTime':  [],
		'Loss':      [],
		'Accuracy':  [],
		'Epochs'  :  0,
		'Cluster' :  ('Test' if cluster == None else cluster)
		}

		if (torch.distributed.is_available()):
			print("DISTRIBUTED Support available")
			backend = "mpi"
#			dist.init_process_group(rank=0, world_size=0)
		else:
			print("PROBLEMS ...")

			# self.global_batch_size = global_batch_size
			# self.batch_size = 0
			# self.batches    = np.arange(self.size)


		rel_speeds = []
		self.devices    = []

		if open == True:
			f = open(sys.argv[3],"r")
			lines = f.readlines()
			f.close()

			i = 0
			for x in lines:
				if   (i == 0): # Number of processes
					P = int(x.split(' ')[1])
				elif (i == 1): # D: size of problem
					D = float(x.split(' ')[1])
				elif (i == 2): # Header
					None
				else:          # Processes lines
					line = x.split('\t')
							# print(line)
				#	rel_speeds.append(float(line[2]) / D)
					self.devices.append(line[8])

				i = i + 1

			rel_speeds = [(1.0 / self.size) for _ in range(self.size)]
			self.balanced = True

		else:


			print("No part_file found: homogeneous distribution.")
			rel_speeds = [(1.0 / self.size) for _ in range(self.size)]
			types = readtype()


			if(types[int(sys.argv[2])] == "cpu"):
				self.devices.append('cpu')
				device = torch.device('cpu')
			else:
				self.devices.append('cuda')
				self.rank_gpu = int(sys.argv[2]) % torch.cuda.device_count()
				print("RANK GPU:", self.rank)
				device = torch.device('cuda')
				torch.cuda.set_device(self.rank_gpu)
			self.balanced = False


		# return np.array(rel_speed), devices
		self.relative_speeds = np.array(rel_speeds)

		# self.relative_speeds, self.devices = getPartInfo(part_file)
		self.relative_speed  = self.relative_speeds[self.rank]
		print(self.devices)
		self.model  = None # CNNet()

		# This must be a parameter to the process
#		self.device = self.devices[0]
		self.device = torch.device('cuda' if (self.devices[self.rank] == 'cuda' and torch.cuda.is_available()) else 'cpu')
#		torch.cuda.set_device(self.rank_gpu)
#		if self.device == torch.device('cuda'):
#			self.rank_gpu = self.rank % torch.cuda.device_count()

		print("ARGV2 is:", (sys.argv[2]))
		print("[",self.rank,"]  Speeds: ", self.relative_speeds, "   D: ", self.relative_speed)
		print("[",self.rank,"]  running in device:", self.device)
		print("[",self.rank,"]  rank GPU: ", self.rank_gpu)


	def getSpeeds (self):
		return self.relative_speeds


	def setData (self, loader):
		self.train_loader = loader.train_loader
		self.val_loader   = loader.val_loader
		self.test_loader  = loader.test_loader
		self.loader = loader


	def setModel (self, model):
		self.model = model
		print("SELF DEVICE:\t", self.device)
		self.model.to(self.device)



	def __update_gradients (self):

		# print("[", self.rank, "/", self.size, "] Updating gradients ... ")
		for param in self.model.parameters():

			a = dist.all_reduce(param.grad.data[torch.isnan(param.grad.data)!=1], op=dist.ReduceOp.SUM, async_op=False)
			#a.wait()

			# Average gradients
			param.grad.data[torch.isnan(param.grad.data)!=1] /= self.size



	def __benchmarkToFile (self, performance):

		model_name = self.model.name
		data_name  = self.loader.name
		balanced   = 'B' if self.balanced == True else 'NB'
		cluster    = performance["Cluster"]

		# Global (joint) data
		if self.rank == 0:

			name = cluster + "_" + model_name + "_" + data_name + "_" + balanced + "_" + str(self.size) + "_" + sys.argv[2] + ".txt"

			f = open("/mnt/shared/jarico/fupermod-latest/het_tests/torch_scripts/" + name, "w+")

			f.write("#Size\t%d\n" % self.size)
			f.write("#BatchSize\t%d\n" % (self.train_loader.batch_size))

			f.write("#Epoch\tTcomp\tTcomm\tTtotal\tLoss\tAccuracy\n")


		tcomp  = torch.FloatTensor(performance["CompTime"])
		tcomm  = torch.FloatTensor(performance["CommTime"])
		ttotal = torch.FloatTensor(performance["EpochTime"])
		loss   = torch.FloatTensor(performance["Loss"])
		acc    = torch.FloatTensor(performance["Accuracy"])


		dist.reduce(tcomp,  0, op=dist.ReduceOp.MAX, async_op=False)
		dist.reduce(tcomm,  0, op=dist.ReduceOp.MAX, async_op=False)
		dist.reduce(ttotal, 0, op=dist.ReduceOp.MAX, async_op=False)

		dist.reduce(loss,   0, op=dist.ReduceOp.MIN, async_op=False)
		dist.reduce(acc,    0, op=dist.ReduceOp.MAX, async_op=False)

		if (self.rank == 0):
			for e in range(performance["Epochs"]):
				f.write("%d\t%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.6f\n" % (e, tcomp[e], tcomm[e], ttotal[e], loss[e], acc[e]))

			f.close()

		# Per replica data
		name_r = cluster + "_" + model_name + "_" + data_name + "_" + balanced + "_" + str(self.size) + "_" + sys.argv[2] + "_" + str(self.rank) + ".txt"
		print("[", self.rank, "/", self.size, "] Opening file: ", name_r, performance["Epochs"])

		f = open("/mnt/shared/jarico/fupermod-latest/het_tests/torch_scripts/" + name_r, "w+")

		f.write("#Size\t%d\n" % self.size)
		f.write("#BatchSize\t%d\n" % (self.train_loader.batch_size))

		f.write("#Epoch\tTcomp\tTcomm\tTtotal\tLoss\tAccuracy\n")
		for e in range(performance["Epochs"]):
			f.write("%d\t%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.6f\n" % (e,
																 performance["CompTime"][e],
																 performance["CommTime"][e],
																 performance["EpochTime"][e],
																 performance["Loss"][e],
																 performance["Accuracy"][e]))

		f.close()



	def benchmark (self):

		# switch to train mode
		model.train()

		start = time.time()

		print("[", self.rank, "/", self.size, "] benchmarking in ", self.hostname)

		epochs = 4
		for e in range(epochs):

			running_loss = 0

			tepoch_start = time.time()
			tcomp = 0.0
			tcomm = 0.0
			tepoch = 0.0

			b = 0

			for images, labels in self.train_loader:
				# print ("[",rank,"]", labels)
				tcomp_start = time.time()
				#				images = images.view(images.shape[0], -1)

				# print("[", self.rank, "] iter: ", b, " of epoch: ", e)
				b = b + 1

				if (self.device == torch.device('cpu')):
				#if (self.device == 'cpu'):
					images = Variable(images)
					labels = Variable(labels)

				else:  # GPU
					images = Variable(images.to(self.device))
					labels = Variable(labels.to(self.device))

				self.model.optimizer.zero_grad() # Avoid accumulate gradients

				output = self.model(images)
				loss = self.model.criterion(output, labels)
				loss.backward();


				tcomm_start = time.time()

				tcomm_end = time.time()
				tcomm += (tcomm_end - tcomm_start)

				self.model.optimizer.step()


				running_loss += loss

				tcomp_end = time.time()
				tcomp += (tcomp_end - tcomp_start)

			tepoch_end = time.time()
			tepoch = tepoch_end - tepoch_start
			tcomp = tcomp - tcomm

			self.performance["Epochs"] += 1
			self.performance["CompTime"].append(tcomp)
			self.performance["CommTime"].append(tcomm)
			self.performance["EpochTime"].append(tepoch)

			self.performance['Loss'].append(running_loss / len(self.train_loader))
			self.performance['Accuracy'].append(0.0) # TBD

			print("[",sys.argv[2], "] ","BatchSize:" , int(int(sys.argv[1])/int(sys.argv[4])),
			"Epoch:   {}/{} -       ".format(self.performance["Epochs"], epochs),
			"Training Loss: {:.3f}  ".format(running_loss / len(self.train_loader)),
			"Time {:.6f} / Comp {:.6f} / Comm {:.6f} ".format(tepoch, tcomp, tcomm))

		end = time.time()
		self.performance['Time'] = end - start
		# self.performance['Loss'] = running_loss / len(self.train_loader)

#		self.__benchmarkToFile(self.performance)



	def run (self):
		pass


	def getPerformance (self):
		return self.performance




################################################################################
##########          MODELS
################################################################################

class CNNet (nn.Module):

	def __init__(self):
		super(CNNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)



class SimpleMNISTModel (nn.Module):

	def __init__(self):
		super(SimpleMNISTModel, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)

		self.optimizer = optim.SGD (self.parameters(), lr=0.003)
		self.criterion = nn.CrossEntropyLoss ()

		self.name = 'SimpleMNISTModel'


	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)



class SimpleCIFAR10Model (nn.Module):

	def __init__(self):
		super(SimpleCIFAR10Model, self).__init__()
		self.conv1 = nn.Conv2d(3, 10, kernel_size=7, padding=3)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=7, padding=3)
		self.fc1 = nn.Linear(20 * 8 * 8, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

		self.optimizer = optim.SGD (self.parameters(), lr=0.003)
		self.criterion = nn.CrossEntropyLoss ()

		self.name = 'SimpleCIFAR10Model'

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 20 * 8 * 8)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return F.log_softmax(x, dim=1)



class SimpleIMAGENETModel (nn.Module):

	def __init__(self):
		super(SimpleIMAGENETModel, self).__init__()
		self.conv1 = nn.Conv2d(3, 10, kernel_size=5, padding=2, stride=2)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=2)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(3920, 2000)
		self.fc2 = nn.Linear(2000, 1000)

		self.optimizer = optim.SGD (self.parameters(), lr=0.003)
		self.criterion = nn.CrossEntropyLoss ()

		self.name = 'SimpleIMAGENETModel'


	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		x = x.view(-1, 20 * 14 * 14)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)




################################################################################
##########          DATASETS
################################################################################


class HeterogeneousDistributedSampler(Sampler):

	def __init__(self, dataset, speeds):

		super(HeterogeneousDistributedSampler, self).__init__(dataset)

		#if not dist.is_available():
		#	raise RuntimeError("Requires distributed package to be available")

		size = 0
		rank = 0

		self.dataset = dataset
		self.size = size
		self.rank = rank
		self.epoch = 0
		self.num_samples = 10000

		print("[", self.rank, "/", self.size, "] Samples: ", self.num_samples, "  speed: ", speeds[self.rank])

		self.starti = 0
		for i in range(0, rank):
			self.starti += int(len(self.dataset) * speeds[i])
		self.endi = self.starti + self.num_samples
		# self.total_size = len(self.dataset)

	def __iter__(self):
		# deterministically shuffle based on epoch
		g = torch.Generator()
		g.manual_seed(self.epoch)
		indices = torch.randperm(len(self.dataset), generator=g).tolist()

		# subsample
		indices = indices[self.starti : self.endi]

		#print("[", self.rank, "/", self.size, "] Indices: ", len(indices), "  from: ", self.starti)

		return iter(indices)

	def __len__(self):
		return self.num_samples

	def set_epoch(self, epoch):
		self.epoch = epoch




class MNISTLoader ():

	def __init__(self, batch_size, speeds):

		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307, ), (0.3081, ))
			])

		# choose the training and test datasets
		train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
		val_data   = None
		test_data  = None

		# From speeds to partitions
		self.partitions = np.rint(speeds * batch_size).astype('int')
		if np.sum(self.partitions) != batch_size:
			self.partitions[np.argmax(self.partitions)] += (batch_size - np.sum(self.partitions))
		# self.partitions.astype(int)
		# print("Partitions: ", self.partitions, "  with type: ", self.partitions[0].dtype)

		self.train_sampler = HeterogeneousDistributedSampler(train_data, speeds)
		# self.train_sampler = DistributedSampler(train_data)

		self.train_loader = torch.utils.data.DataLoader(train_data,
														batch_size  = int(sys.argv[1]),
														num_workers = 4,
														shuffle     = (self.train_sampler is None),
														pin_memory  = False,
														drop_last   = False, # True??
														sampler     = self.train_sampler)

		self.val_loader  = None
		self.test_loader = None

		self.name = 'MNISTLoader'

		# return self.train_loader, self.val_loader, self.test_loader


class CIFAR10Loader ():

	def __init__(self, batch_size, speeds):

		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])

		# choose the training and test datasets
		train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
		val_data   = None
		test_data  = None

		# From speeds to partitions
		self.partitions = np.rint(speeds * batch_size).astype('int')
		if np.sum(self.partitions) != batch_size:
			self.partitions[np.argmax(self.partitions)] += (batch_size - np.sum(self.partitions))


		self.train_sampler = HeterogeneousDistributedSampler(train_data, speeds)
		# self.train_sampler = DistributedSampler(train_data)

		self.train_loader = torch.utils.data.DataLoader(train_data,
														batch_size  = int(int(sys.argv[1])/int(sys.argv[4])),
														num_workers = 0,
														shuffle     = (self.train_sampler is None),
														pin_memory  = False,
														drop_last   = False,
														sampler     = self.train_sampler)

		self.val_loader  = None
		self.test_loader = None

		self.name = 'CIFAR10Loader'
		# return self.train_loader, self.val_loader, self.test_loader



class IMAGENETLoader ():

	def __init__(self, batch_size, speeds):

		transform = transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])
			])

		# choose the training and test datasets
		train_data = datasets.ImageFolder(root='data', train=True, download=True, transform=transform)
		val_data   = None
		test_data  = None

		# From speeds to partitions
		self.partitions = np.rint(speeds * batch_size).astype('int')
		if np.sum(self.partitions) != batch_size:
			self.partitions[np.argmax(self.partitions)] += (batch_size - np.sum(self.partitions))
		# self.partitions.astype(int)
		# print("Partitions: ", self.partitions, "  with type: ", self.partitions[0].dtype

		self.train_sampler = HeterogeneousDistributedSampler(train_data, speeds)
		# self.train_sampler = DistributedSampler(train_data)

		self.train_loader = torch.utils.data.DataLoader(train_data,
														batch_size  = int(self.partitions[dist.get_rank()]),
														num_workers = 0,
														shuffle     = (self.train_sampler is None),
														pin_memory  = False,
														drop_last   = True,
														sampler     = self.train_sampler)

		self.val_loader  = None
		self.test_loader = None

		self.name = 'IMAGENETLoader'


################################################################################
##########          MAIN
################################################################################

if __name__ == "__main__":



	# 1) Create Replicas. Read info from part_file with speeds and devices.
	p      = Replica            (None, open=False)
	# p      = Replica            ("-")
	loader = CIFAR10Loader      (int(sys.argv[1]), p.getSpeeds())
	model  = SimpleCIFAR10Model ()

	p.setData  (loader)
	p.setModel (model)

	p.benchmark ()
#	print("[", p.rank, "/", p.size, "] ", p.getPerformance())

