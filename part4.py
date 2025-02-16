####################################################################
# CEN/CSE 524: Machine Learning Acceleration
# Instructor: Dr. Aman Arora
# Spring 2025
# TA: Kaustubh Mhatre
####################################################################
import os
import numpy as np
from transformers import BertConfig, BertForQuestionAnswering, BertTokenizer
from transformers.data.processors.squad import SquadV1Processor
from transformers import squad_convert_examples_to_features
from torch.utils.data import DataLoader
import torch
import time

cache_dir = os.path.join(".", "cache_models")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

predict_file_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
predict_file = os.path.join(cache_dir, "dev-v1.1.json")
if not os.path.exists(predict_file):
    import wget

    print("Start downloading predict file.")
    wget.download(predict_file_url, predict_file)
    print("Predict file downloaded.")


####################################################################
# define the BERT base model and its max seq length
model_name_or_path = "bert-base-cased"
max_seq_length = 128
doc_stride = 128
max_query_length = 64


# Total samples to inference. It shall be large enough to get stable latency measurement.
total_samples = 100
device = "cpu"

####################################################################
# Load pretrained model and tokenizer
config_class, model_class, tokenizer_class = (
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
)
config = config_class.from_pretrained(model_name_or_path, cache_dir=cache_dir)
# print(config)
tokenizer = tokenizer_class.from_pretrained(
    model_name_or_path, do_lower_case=True, cache_dir=cache_dir
)
model = model_class.from_pretrained(
    model_name_or_path, from_tf=False, config=config, cache_dir=cache_dir
)

# load some examples
processor = SquadV1Processor()
examples = processor.get_dev_examples(None, filename=predict_file)


features, dataset = squad_convert_examples_to_features(
    examples=examples[:total_samples],  # convert just enough examples for this notebook
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    doc_stride=doc_stride,
    max_query_length=max_query_length,
    is_training=False,
    return_dataset="pt",
    threads=1,
)


# Parameters
batch_size = 8  # Modify the batch size as needed TODO: Change this variable to change the batchsize

# Create a DataLoader
data_loader = DataLoader(
    dataset, batch_size=batch_size, shuffle=False  # Disable shuffle if order matters
)

latency = []

output_dir = os.path.join(".", "onnx_models")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
export_model_path = os.path.join(output_dir, "bert-base-cased-squad.onnx")


# device = torch.device("cuda")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Get the first example data to run the model
data = dataset[0]
inputs = {
    "input_ids": data[0].to(device).reshape(1, max_seq_length),
    "attention_mask": data[1].to(device).reshape(1, max_seq_length),
    "token_type_ids": data[2].to(device).reshape(1, max_seq_length),
}

# Set model to inference mode,
model.eval()
model.to(device)

test_iterator = iter(data_loader)
batch = next(test_iterator)

input_ids = batch[0].to(device)  # Shape: (batch_size, max_seq_length)
attention_mask = batch[1].to(device)  # Shape: (batch_size, max_seq_length)
token_type_ids = batch[2].to(device)  # Shape: (batch_size, max_seq_length)

inputs = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "token_type_ids": token_type_ids,
}

# Measure inference time
start = time.time()
# Run for 50 iterations to remove measurement overhead
for i in range(1):
    outputs = model(**inputs)
end = time.time()
latency = (end - start) / 50
print("Latency : ", latency)
