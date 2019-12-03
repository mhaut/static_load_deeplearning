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


class Replica ():

	def __init__ (self, part_file="part.dist", balance=1, cluster = None):

		self.size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
		self.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
		self.rank_gpu = None
		self.hostname = socket.gethostname()

		self.performance = {
		'Time':      0.0,
		'EpochTime': [],
		'CompTime':  [],
		'CommTime':  [],
		'Epochs'  :  0,
		'Cluster' :  ('Test' if cluster == None else cluster),
		'Train_loss':[],
		'Test_loss':[],
		'TOP_1':[],
		'TOP_5':[]
		}

		if (torch.distributed.is_available()):
			print("DISTRIBUTED Support available")
			backend = "mpi"
			dist.init_process_group(backend, rank=self.rank, world_size=self.size)
		else:
			print("PROBLEMS ...")

			# self.global_batch_size = global_batch_size
			# self.batch_size = 0
			# self.batches    = np.arange(self.size)

		rel_speeds = []
		self.devices    = []
		speeds = []
		batches = []

		try:
			f = open(part_file,"r")
			lines = f.readlines()
			f.close()

			for i,x in enumerate(lines):
				if   (i == 0): # Number of processes
					P = int(x.split(' ')[1])
				elif (i == 1): # D: size of problem
					D = float(x.split(' ')[1])
				elif (i == 2): # Header
					pass
				else:          # Processes lines
					line = x.split('\t')
							# print(line)
					rel_speeds.append(float(line[2]) / D)
					self.devices.append(line[8])
					if balance == 1:
						batches.append(int(int(x.split('\t')[2]) / 3136))
					else:
						batches.append(int(int(sys.argv[2])/self.size))

			if sum(batches) != int(sys.argv[2]) and balance == 1:
				batches[np.argmax(batches)] = (int(sys.argv[2]) - sum(batches)) + max(batches)

			self.balanced = True
			self.relative_batches = np.array(batches)
		except:
			print("No part_file found: homogeneous distribution.")
			#rel_speeds = [(1.0 / self.size) for _ in range(self.size)]
			#self.devices = ['cpu' for _ in range(self.size)]

			#self.balanced = False

		if balance == 0:
			self.balanced = False
			self.relative_batches = np.array(batches)
			rel_speeds = [(1.0 / self.size) for _ in range(self.size)]

		# return np.array(rel_speed), devices
		self.relative_speeds = np.array(rel_speeds)
		self.relative_batches = np.array(batches)

		# self.relative_speeds, self.devices = getPartInfo(part_file)
		self.relative_speed  = self.relative_speeds[self.rank]

		self.model  = None # CNNet()

		# This must be a parameter to the process
		#self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.device = torch.device('cuda' if (self.devices[self.rank].rstrip() == 'gpu' and torch.cuda.is_available()) else 'cpu')

		if self.device == torch.device('cuda'):
			self.rank_gpu = self.rank % torch.cuda.device_count()
			print("rank gpu:\t", self.rank_gpu)
			torch.cuda.set_device(self.rank_gpu)

		print("[",self.rank,"]  Speeds: ", self.relative_speeds, "   D: ", self.relative_speed)
		print("[",self.rank,"]  running in device:", self.device)
		print("[",self.rank,"]  rank GPU: ", self.rank_gpu)


	def getSpeeds (self):
		return self.relative_speeds

	def getBatches (self):
		return self.relative_batches

	def setData (self, loader):
		self.train_loader = loader.train_loader
		self.val_loader   = loader.val_loader
		self.test_loader  = loader.test_loader
		self.loader = loader


	def setModel (self, model):
		self.model = model
		self.model.to(self.device)



	def __update_gradients (self):

		# print("[", self.rank, "/", self.size, "] Updating gradients ... ")
		#print("Entro\t", dist.get_rank())

		for param in self.model.parameters():
#			a = dist.all_reduce(param.grad.data[torch.isnan(param.grad.data)!=1], op=dist.ReduceOp.SUM, async_op=True)
			dist.all_reduce(param.grad.data[torch.isnan(param.grad.data)!=1], op=dist.ReduceOp.SUM, async_op=False)
			#print("Entra wait\t", self.rank)
#			a.wait()
			#print("Sale wait\t", self.rank)
			# Average gradient 
			param.grad.data[torch.isnan(param.grad.data)!=1] /= self.size
		#print("Salgo:\t", dist.get_rank())


	def __benchmarkToFile (self, performance):

		
		data_name  = self.loader.name
		balanced   = 'B' if self.balanced == True else 'NB'
		cluster    = performance["Cluster"]

		# Global (joint) data
		if self.rank == 0:

			name = "TotalTime_" + balanced + "_" + str(self.rank) + "_" + sys.argv[2] + ".txt"

			f = open("/home/jarico/ws/t-lop/hetbatch/results" + "/" + sys.argv[5] + "_" + sys.argv[6] + "_" + sys.argv[7] + "/" + name, "w+")

			f.write("#Size\t%d\n" % self.size)
			f.write("#BatchSize\t%d\n" % (self.loader.batch_size))

			f.write("#Epoch\tTcomp\tTcomm\tTtotal\tLoss\tAccuracy\n")


		tcomp  = torch.FloatTensor(performance["CompTime"])
		tcomm  = torch.FloatTensor(performance["CommTime"])
		ttotal = torch.FloatTensor(performance["EpochTime"])
		train_loss = torch.FloatTensor(performance["Train_loss"])
		test_loss  = torch.FloatTensor(performance["Test_loss"])
		top1    = torch.FloatTensor(performance["TOP_1"])
		top5   = torch.FloatTensor(performance["TOP_5"])

		dist.reduce(tcomp,  0, op=dist.ReduceOp.MAX, async_op=False)
		dist.reduce(tcomm,  0, op=dist.ReduceOp.MAX, async_op=False)
		dist.reduce(ttotal, 0, op=dist.ReduceOp.MAX, async_op=False)

		dist.reduce(train_loss,   0, op=dist.ReduceOp.MIN, async_op=False)
		dist.reduce(test_loss,   0, op=dist.ReduceOp.MIN, async_op=False)
		dist.reduce(top1,    0, op=dist.ReduceOp.MAX, async_op=False)
		dist.reduce(top5,    0, op=dist.ReduceOp.MAX, async_op=False)

		if (self.rank == 0):
			for e in range(performance["Epochs"]):
				f.write("%d\t%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.6f\n" % (e, tcomp[e], tcomm[e], ttotal[e], train_loss[e], test_loss[e], top1[e], top5[e]))

			f.close()

		# Per replica data
		name_r = "ReplicaTime_" + balanced + "_" + str(self.rank) + "_" + sys.argv[2] + ".txt"
		print("[", self.rank, "/", self.size, "] Opening file: ", name_r, performance["Epochs"])

		f = open("/home/jarico/ws/t-lop/hetbatch/results" + "/" + sys.argv[5] + "_" + sys.argv[6] + "_" + sys.argv[7] + "/" + name_r, "w+")

		f.write("#Size\t%d\n" % self.size)
		f.write("#BatchSize\t%d\n" % (self.train_loader.batch_size))

		f.write("#Epoch\tTcomp\tTcomm\tTtotal\tTrain_loss\tTest_loss\tTOP_1\tTOP_5\n")
		for e in range(performance["Epochs"]):
			f.write("%d\t%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.6f\n" % (e,
																 performance["CompTime"][e],
																 performance["CommTime"][e],
																 performance["EpochTime"][e],
																 performance["Train_loss"][e],
																 performance["Test_loss"][e],
																 performance["TOP_1"][e],
																 performance["TOP_5"][e]))

		f.close()




	def get_acc(self, output, target, topk=(1,)):
		with torch.no_grad():
			maxk = max(topk)
			batch_size = target.size(0)

			_, pred = output.topk(maxk, 1, True, True)
			pred = pred.t()
			correct = pred.eq(target.view(1, -1).expand_as(pred))

			res = []
			for k in topk:
				correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
				res.append(correct_k.mul_(100.0 / batch_size))
			return res


	def validate(self, criterion):
		# switch to evaluate mode
		self.model.eval()

		with torch.no_grad():
			for i, (images, labels) in enumerate(self.test_loader):
				if (self.device == torch.device('cpu')):
					images = Variable(images)
					labels = Variable(labels)
				else:  # GPU
					images = Variable(images.to(self.device))
					labels = Variable(labels.to(self.device))

				output = self.model(images)

				val_loss = criterion(output, labels).item()
				acc1, acc5 = self.get_acc(output, labels, topk=(1, 5))

		return acc1[0], acc5[0], val_loss



	def benchmark(self, with_test=False, epochs = 10):

		# switch to train mode
		self.model.train()

		print("[", self.rank, "/", self.size, "] benchmarking in ", self.hostname, "Len:\t", len(self.train_loader), self.relative_batches[self.rank])

		start = time.time()


		#optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)
		optimizer = optim.Adam(self.model.parameters())
		criterion = nn.CrossEntropyLoss()
		for e in range(epochs):
			running_loss = 0
			tepoch_start = time.time()
			tcomp = 0.0
			tcomm = 0.0
			tepoch = 0.0

			for images, labels in self.train_loader:
				# print ("[",rank,"]", labels)
				tcomp_start = time.time()
				#				images = images.view(images.shape[0], -1)

				# print("[", self.rank, "] iter: ", b, " of epoch: ", e)
				#b = b + 1

				if (self.device == torch.device('cpu')):
				#if (self.device == 'cpu'):
					images = Variable(images)
					labels = Variable(labels)
					# print("DEVICE TO CPU: ", self.rank, self.device)
				else:  # GPU
					images = Variable(images.to(self.device))
					labels = Variable(labels.to(self.device))
					# images = Variable(images.cuda(self.rank % torch.cuda.device_count()))
					# labels = Variable(labels.cuda(self.rank % torch.cuda.device_count()))
					# print("DEVICE TO GPU: ", self.rank % torch.cuda.device_count(), self.device, self.rank_gpu, self.hostname)

				#self.model.optimizer.zero_grad() # Avoid accumulate gradients
				optimizer.zero_grad()
				
				output = self.model(images)
				loss = criterion(output, labels)
				loss.backward();

				#for W in self.model.parameters():
					#mask = (W.data.abs() > 1.e-32).float()
					#W.data.copy_(W.data * mask) # W - eta*g + A*gdl_eps
				## Manually zero the gradients after updating weights
				#self.model.zero_grad()

#				tcomp_end = time.time()
#				tcomp += (tcomp_end - tcomp_start)

				tcomm_start = time.time()
				self.__update_gradients()
				tcomm_end = time.time()
				tcomm += (tcomm_end - tcomm_start)

#				self.model.optimizer.step()
				optimizer.step()

#				tcomm_end = time.time()
#				tcomm += (tcomm_end - tcomm_start)

				running_loss += loss.item()

				tcomp_end = time.time()
				tcomp += (tcomp_end - tcomp_start)
#			torch.cuda.synchronize()
			tepoch_end = time.time()
			tepoch = tepoch_end - tepoch_start
			tcomp = tepoch - tcomm

			self.performance["Epochs"] += 1
			self.performance["CompTime"].append(tcomp)
			self.performance["CommTime"].append(tcomm)
			self.performance["EpochTime"].append(tepoch)

			self.performance['Train_loss'].append(running_loss / len(self.train_loader))


			print("[", self.rank, "/", self.size, "] ",
			"Epoch:   {}/{} -       ".format(self.performance["Epochs"], epochs),
			"Training Loss: {:.3f}  ".format(running_loss / len(self.train_loader)),
			"Time {:.6f} / Comp {:.6f} / Comm {:.6f}".format(tepoch, tcomp, tcomm), end=" / " if with_test else "\n")

			if with_test:
				acc1, acc5, test_loss = self.validate(criterion)
				self.model.train()
				self.performance['TOP_1'].append(acc1)
				self.performance['TOP_5'].append(acc5)
				self.performance['Test_loss'].append(test_loss)

				print("Test Loss: {:.3f} ".format(test_loss), "/ Top-1 {:.2f} / Top-5 {:.2f} ".format(acc1, acc5))
				self.performance['Test_loss'].append(test_loss)
			else:
				self.performance['Test_loss'].append(-1)
				self.performance['TOP_1'].append(-1)
				self.performance['TOP_5'].append(-1)


		end = time.time()
		self.performance['Time'] = end - start
		# self.performance['Loss'] = running_loss / len(self.train_loader)

		self.__benchmarkToFile(self.performance)



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
		return x



class SimpleMNISTModel (nn.Module):

	def __init__(self):
		super(SimpleMNISTModel, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)
		self.name = 'SimpleMNISTModel'


	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		return x



class SimpleCIFAR10Model (nn.Module):

	def __init__(self):
		super(SimpleCIFAR10Model, self).__init__()
		self.conv1 = nn.Conv2d(3, 10, kernel_size=7, padding=3)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=7, padding=3)
		self.fc1 = nn.Linear(20 * 8 * 8, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)
		self.name = 'SimpleCIFAR10Model'

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 20 * 8 * 8)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x



class SimpleIMAGENETModel (nn.Module):

	def __init__(self):
		super(SimpleIMAGENETModel, self).__init__()
		self.conv1 = nn.Conv2d(3, 10, kernel_size=5, padding=2, stride=2)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=2)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(3920, 2000)
		self.fc2 = nn.Linear(2000, 1000)
		self.name = 'SimpleIMAGENETModel'


	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		x = x.view(-1, 20 * 14 * 14)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		return x




################################################################################
##########          DATASETS
################################################################################


class HeterogeneousDistributedSampler(Sampler):

	def __init__(self, dataset, speeds, batches, test=False):

		super(HeterogeneousDistributedSampler, self).__init__(dataset)

		if not dist.is_available():
			raise RuntimeError("Requires distributed package to be available")

		self.dataset = dataset
		self.size = dist.get_world_size()
		self.rank = dist.get_rank()
		self.epoch = 0
		
		self.num_samples = int(len(self.dataset) * speeds[self.rank])
		
		
		
		#self.num_samples = int (speeds[rank] / np.sum(speeds))
	
		#for i in range(self.size):
		#	while(int(self.num_samples/batches[self.rank]) > int(len(self.dataset) * speeds[i]/batches[i])):
		#		self.num_samples -= batches[self.rank]
		
		#while (self.num_samples % batches[self.rank] != 0): self.num_samples -= 1

		if test:
			print("[", self.rank, "/", self.size, "] Test Samples: ", self.num_samples, "  speed: ", speeds[self.rank])
		else:
			print("[", self.rank, "/", self.size, "] Samples: ", self.num_samples, "  speed: ", speeds[self.rank])

		self.starti = 0
		for i in range(self.rank): self.starti += int(len(self.dataset) * speeds[i])
		self.endi = self.starti + self.num_samples

	def __iter__(self):
		# deterministically shuffle based on epoch
		g = torch.Generator()
		g.manual_seed(self.epoch)
		indices = torch.randperm(len(self.dataset), generator=g).tolist()
		# subsample
		indices = indices[self.starti : self.endi]
		self.epoch += 1
		return iter(indices)

	def __len__(self):
		return self.num_samples






class MNISTLoader ():

	def __init__(self, batch_size, speeds, batches):

		transform_train = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307, ), (0.3081, ))
			])
		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307, ), (0.3081, ))
			])


		# choose the training and test datasets
		train_data = datasets.MNIST(root='/home/jarico/ws/t-lop/hetbatch/data', train=True, download=True, transform=transform_train)
		test_data  = datasets.MNIST(root='/home/jarico/ws/t-lop/hetbatch/data', train=False, download=True, transform=transform_test)
		val_data   = None

		# From speeds to partitions
		self.partitions = np.rint(speeds * batch_size).astype('int')
		if np.sum(self.partitions) != batch_size:
			self.partitions[np.argmax(self.partitions)] += (batch_size - np.sum(self.partitions))

		# self.partitions.astype(int)
		# print("Partitions: ", self.partitions, "  with type: ", self.partitions[0].dtype)

		self.train_sampler = HeterogeneousDistributedSampler(train_data, speeds, batches)
		self.test_sampler  = HeterogeneousDistributedSampler(test_data, speeds, batches, test=True)

		if len(self.train_sampler.dataset) * speeds[dist.get_rank()] < 0.5:
			drop = False
		else:
			drop = True
		print(dist.get_rank(), "---------", batches[dist.get_rank()])
		# self.train_sampler = DistributedSampler(train_data)

		self.train_loader = torch.utils.data.DataLoader(train_data,
													batch_size  = int(self.partitions[dist.get_rank()]),
													num_workers = 0,
													shuffle     = (self.train_sampler is None),
													pin_memory  = False,
													drop_last   = drop,
													sampler     = self.train_sampler)
		self.test_loader = torch.utils.data.DataLoader(test_data,
													batch_size  = int(batches[dist.get_rank()]),
													num_workers = 0,
													shuffle     = False,
													pin_memory  = False,
													drop_last   = False,
													sampler     = self.test_sampler)
		self.val_loader  = None

		self.name = 'MNISTLoader'

		self.rank_batch_size = int(self.partitions[dist.get_rank()])
		#self.batch_size = batch_sizesampler     = self.train_sampler
		self.batch_size = batch_size
		print("[", dist.get_rank(), "] Num. Samples: ", self.train_sampler.num_samples, "  Batch size: ", self.rank_batch_size, " Iters: ", self.train_sampler.num_samples/self.rank_batch_size)


class CIFAR10Loader ():

	def __init__(self, batch_size, speeds, batches):
		print('==> Preparing data..')

		transform_train = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])

		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])

		# choose the training and test datasets
		train_data = datasets.CIFAR10(root='/home/jarico/ws/t-lop/hetbatch/data', train=True, download=True, transform=transform_train)
		val_data   = None
		test_data = datasets.CIFAR10(root='/home/jarico/ws/t-lop/hetbatch/data', train=False, download=True, transform=transform_test)


		# From speeds to partitions
		self.partitions = np.rint(speeds * batch_size).astype('int')
		if np.sum(self.partitions) != batch_size:
			self.partitions[np.argmax(self.partitions)] += (batch_size - np.sum(self.partitions))
		# self.partitions.astype(int)
		# print("Partitions: ", self.partitions, "  with type: ", self.partitions[0].dtype)

		self.train_sampler = HeterogeneousDistributedSampler(train_data, speeds, batches)
		self.test_sampler  = HeterogeneousDistributedSampler(test_data, speeds, batches, test=True)
		# self.train_sampler = DistributedSampler(train_data)
		print("==================" , batches)

		if len(self.train_sampler.dataset) * speeds[dist.get_rank()] < 0.5:
			drop = False
		else:
			drop = True

		self.train_loader = torch.utils.data.DataLoader(train_data,
													batch_size  = int(batches[dist.get_rank()]),
													num_workers = 0,
													shuffle     = (self.train_sampler is None),
													pin_memory  = False,
													drop_last   = drop,
													sampler     = self.train_sampler)

		self.test_loader = torch.utils.data.DataLoader(test_data,
													batch_size  = int(batches[dist.get_rank()]),
													num_workers = 0,
													shuffle     = False,
													pin_memory  = False,
													drop_last   = False,
													sampler     = self.test_sampler)

		self.val_loader  = None
		

		self.name = 'CIFAR10Loader'
		# return self.train_loader, self.val_loader, self.test_loader

		self.rank_batch_size = int(self.partitions[dist.get_rank()])
		#self.batch_size = batch_sizesampler     = self.train_sampler
		self.batch_size = batch_size
		print("[", dist.get_rank(), "] Num. Samples: ", self.train_sampler.num_samples, "  Batch size: ", self.rank_batch_size, " Iters: ", self.train_sampler.num_samples/self.rank_batch_size)


class CIFAR10ResnetLoader ():

	def __init__(self, batch_size, speeds, batches):
		print('==> Preparing data..')

		transform_train = transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.RandomCrop(32, 4),
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])

		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])


                # choose the training and test datasets
		train_data = datasets.CIFAR10(root='/home/jarico/ws/t-lop/hetbatch/data', train=True, download=True, transform=transform_train)
		val_data   = None
		test_data = datasets.CIFAR10(root='/home/jarico/ws/t-lop/hetbatch/data', train=False, download=True, transform=transform_test)


                # From speeds to partitions
		self.partitions = np.rint(speeds * batch_size).astype('int')
		if np.sum(self.partitions) != batch_size:
			self.partitions[np.argmax(self.partitions)] += (batch_size - np.sum(self.partitions))

		self.train_sampler = HeterogeneousDistributedSampler(train_data, speeds, batches)
		self.test_sampler  = HeterogeneousDistributedSampler(test_data, speeds, batches)
                # self.train_sampler = DistributedSampler(train_data)
       
		self.train_loader = torch.utils.data.DataLoader(train_data,
												batch_size  = int(batches[dist.get_rank()]),
												num_workers = 0,
												shuffle     = (self.train_sampler is None),
												pin_memory  = False,
												drop_last   = False if len(self.train_sampler.dataset) * speeds[dist.get_rank()] < 0.5 else True,
												sampler     = self.train_sampler)

		self.test_loader = torch.utils.data.DataLoader(test_data,
												batch_size  = int(batches[dist.get_rank()]),
												num_workers = 0,
												shuffle     = False,
												pin_memory  = False,
												drop_last   = False if len(self.train_sampler.dataset) * speeds[dist.get_rank()] < 0.5 else True, 													sampler     = self.test_sampler)

		self.val_loader  = None


		self.name = 'RS_SimpleModel_CIFAR10'
                # return self.train_loader, self.val_loader, self.test_loader

		self.rank_batch_size = int(self.partitions[dist.get_rank()])
                #self.batch_size = batch_sizesampler     = self.train_sampler
		self.batch_size = batch_size
		print("[", dist.get_rank(), "] Num. Samples: ", self.train_sampler.num_samples, "  Batch size: ", self.rank_batch_size, " Iters: ", self.train_sampler.num_samples/self.rank_batch_size)


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
		train_data = datasets.ImageFolder(root='/mnt/shared/data', train=True, download=True, transform=transform)
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
														batch_size  = int(self.partitions[dist.get_rank()]),
														num_workers = 0,
														shuffle     = (self.train_sampler is None),
														pin_memory  = False,
														drop_last   = True,
														sampler     = self.train_sampler)

		self.val_loader  = None
		self.test_loader = None

		self.name = 'IMAGENETLoader'

		self.rank_batch_size = int(self.partitions[dist.get_rank()])
		self.batch_size = batch_size
		print("[", dist.get_rank(), "] Num. Samples: ", self.train_sampler.num_samples, "  Batch size: ", self.rank_batch_size, " Iters: ", self.train_sampler.num_samples/self.rank_batch_size)



################################################################################
##########          MAIN
################################################################################

if __name__ == "__main__":

	# [part_file]: what is the partition file (resulting from partitioner in FuPerMod).
	#               (default: part.dist in .)
	# [balanced]:  True  => use load balancing (FuPerMod)
	#		       False => homogeneously distribute load (default)

	part_file  = sys.argv[1]
	batch_size = int(sys.argv[2]) # Global batch size
	#balanced   = bool(int(sys.argv[3]))
	
	######## PARAMETRIZAR ESTO

	dset = sys.argv[5]
	show_test = False
	##########################

	print("part_file: ", part_file)
	print("Global batch_size: ", batch_size)

	# 1) Create Replicas. Read info from part_file with speeds and devices.
	p      = Replica            (part_file, balance=int(sys.argv[3]))
	# p      = Replica            ("-")
	if dset == "SimpleModel_CIFAR10":
		loader = CIFAR10Loader      (batch_size, p.getSpeeds(), p.getBatches())
		model  = SimpleCIFAR10Model ()
		num_epochs = 10

	elif dset == "RS_SimpleModel_CIFAR10":
		loader = CIFAR10ResnetLoader(batch_size, p.getSpeeds(), p.getBatches())
		import resnet
		
		if int(sys.argv[7]) == 20:
			model  = resnet.resnet20()
		if int(sys.argv[7]) == 32:
			model  = resnet.resnet32()
		if int(sys.argv[7]) == 44:
			model  = resnet.resnet44()
		if int(sys.argv[7]) == 56:
			model  = resnet.resnet56()
		if int(sys.argv[7]) == 110:
			model  = resnet.resnet110()

		num_epochs = 7

	elif dset == "MNIST":
		loader = MNISTLoader      (batch_size, p.getSpeeds(), p.getBatches())
		model  = SimpleMNISTModel ()
		num_epochs = 10
	else:
		print("DATASET NOT FOUND")

	p.setData  (loader)
	p.setModel (model)

	p.benchmark (with_test=show_test, epochs=num_epochs)
	print("[", p.rank, "/", p.size, "] ", p.getPerformance())
