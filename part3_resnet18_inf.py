####################################################################
# CEN/CSE 524: Machine Learning Acceleration
# Instructor: Dr. Aman Arora
# Spring 2025
# TA: Kaustubh Mhatre
####################################################################
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import warnings
import sys

warnings.filterwarnings("ignore")
from torchvision import models, transforms
import time


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if not torch.cuda.is_available():
    print("No GPU found")
    sys.exit(1)
print(f"Using {device} for inference")

####################################################################
# Download a pretrained model for resnet18
resnet18 = models.resnet18(pretrained=True)
utils = torch.hub.load(
    "NVIDIA/DeepLearningExamples:torchhub", "nvidia_convnets_processing_utils"
)

resnet18.eval().to(device)

####################################################################
# Download images and batch them up for inference. You can either add
# image urls below to increase the batch size or you have images from
# other sources

batch_size = (
    128  # Inference batch size TODO: Change this variable to change the batchsize
)


uris = [
    "http://images.cocodataset.org/test-stuff2017/000000024309.jpg",
]

x_uri_bs = [uris[0] for i in range(batch_size)]

batch = torch.cat([utils.prepare_input_from_uri(uri) for uri in x_uri_bs]).to(device)

total_time = 0
with torch.no_grad():
    # Run 100 time to remove any measurement/system overhead
    start_time = time.time()
    for i in range(100):
        output = torch.nn.functional.softmax(resnet18(batch), dim=1)
    end_time = time.time()

    total_time = (end_time - start_time) / 100

print("Inference time for batchsize {} is {}".format(batch_size, total_time))

# Uncomment the below line if you want to see the prediction output
# results = utils.pick_n_best(predictions=output, n=5)
