# SafeDeep

**SafeDeep** is a scalable robustness verification framework for deep neural networks (DNNs).

SafeDeep gets a neural network model and a dataset for verification. The dataset should be a mat file (v4 (Level 1.0), 
v6, and v7 to 7.2 .mat files are supported.) consisting of 1-D signals as inputs and labels. The neural network model 
should be a binary classifier in h5 format containing dense layers with ReLU activation functions for all the hidden 
layers. _Delta_ specifies the L∞ based perturbation, and _use_lp_ determines whether the framework should use 
layer-by-layer bound refining.

## Requirements 

Gurobi's Python Interface, Python 3.9 or higher, TensorFlow 2.10 or higher.


## Usage
```python
python main.py --net_name <path to the network file> --dataset <path to the dataset file> --delta <float between 0 and 1> --use_lp <True/False> 
```

### Example

```python
python main.py --net_name ./Model/model_patient_01.h5 --dataset ./Dataset/patient_01.mat --delta 0.01 --use_lp True 
```

### The paper results
To replicate the paper's results, run the following code. Note that delta can be 0.005, 0.01, 0.02, and 0.04.

```python
python main.py --net_name ./Model/model_all_patients.h5 --dataset ./Dataset/all_patients.mat --delta 0.005 --use_lp True 
```