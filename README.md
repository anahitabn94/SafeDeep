# SafeDeep

**SafeDeep** is a scalable **robustness verification framework** for deep neural networks (DNNs).  
It analyzes whether a given network’s predictions remain **provably correct** under small input perturbations.

## Features

- Verifies the robustness of **binary classifiers** with dense ReLU layers.
- Supports **layer-by-layer bound refinement** using Gurobi MILP for tighter verification.
- Works with **MATLAB `.mat` datasets** containing 1-D signals and corresponding labels.
- Configurable perturbation parameter (`delta`) to specify the magnitude of input uncertainty.

## Requirements

- **Python** 3.9+
- **TensorFlow** 2.10+
- **Gurobi** with Python interface
- Other dependencies:

```bash
pip install numpy scipy pytictoc tqdm
```

## Installation

Clone the repository:

```bash
git clone https://github.com/anahitabn94/SafeDeep.git
cd SafeDeep
```

## Usage

Run the main script:

```bash
python main.py --network <path_to_network_file> --dataset <path_to_dataset_file> --delta <float_between_0_and_1> --lp <True/False>
```

### Paper Results
To replicate the paper's results, run the following code. Note that delta can be 0.005, 0.01, 0.02, and 0.04.

```bash
python main.py --network ./models/model_all_patients.h5 --dataset ./datasets/all_patients.mat --delta 0.005 --lp True 
```

## Output

After running the verification, the framework prints:

Total samples\
Correctly classified samples\
Provably robust\
Elapsed time total

## Notes

- **Dataset format**: `.mat` file with fields:  
  - `dataset`: shape `(num_samples, num_features)`  
  - `label`: shape `(num_samples, 1)` with values `0/1`  

- **Model requirements**: Binary classification model with:  
  - Dense layers only  
  - ReLU activations for hidden layers
