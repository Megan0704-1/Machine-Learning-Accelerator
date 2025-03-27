Machine Learning Acceleration Lab 4 : Quantization and Pruning
==============================================================

In this lab, you will learn about quantization and pruning.

# Setting up your environment
Please request a machine on SOL with following specifications.

- Cores = 8
- Memory = 32 GB
- GPU = (Any GPU is fine)

Create a virtual environment either using python venv, virtualenvwrapper or conda based on your comfort. In this example, we will see how to create a python venv setup. 

To create a virtual environment, we use the following code on the terminal:

    python3 -m venv my_venv python==3.x

Where x can be anywhere in the range of 8 to 12 to ensure stable execution. Once the environment has been created, we activate it using:

    source my_venv/bin/activate

The environment is independent of the lab project and can be initialized anywhere on your system. Once inside the environment, navigate back to the project repository and enter the ``Projects`` folder. Here, you will find a file called `venv_requirements.txt`. This is the list of libraries required to run this project. To install them, execute the following:

    pip install -r venv_requirements.txt 

This should setup your environment completely. Now launch jupyter lab by typing it in on the terminal and you can begin executing your code.

    jupyter-notebook 

This will print a link on the terminal. Open that link in the browser. Browse to the jupyter notebooks and follow the instructions.

## Part1 (1_quantization.ipynb) 50 pts
The first part deals with quantization. You will be using a very small CNN. The goal is to quantize the model from float32 to int8.
Go through this notebook and follow the directions. Complete the TODOs in the notebook and run the cells to observe the outputs.

## Part2 (2_pruning.ipynb) 50 pts
In the second part of the lab, you will experiment with pruning. In this part, you will be using another DNN (VGG).
Go through this notebook and follow the directions. Complete the TODOs in the notebook and run the cells to observe the outputs.

## What to submit?
Create a zip file containing the following files:
- Updated src/quant.py
- Updated 1_quantization.ipynb (with cell outputs) 
- Updated 2_pruning.ipynb (with cell outputs)

Note that you will be submitting these files after completing the TODOs mentioned in the files.
**Submitting the notebooks without the cell output will lead to a ZERO grade.**
