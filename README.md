# CoDET-Attack 

![Python](https://img.shields.io/badge/python-3.11-blue)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![License](https://img.shields.io/badge/license-MIT-green)

Adversarial attack against tree ensemble models (XGBoost) on tabular data with feature constraints.
An SMT solver (Z3) finds initial adversarial points that satisfy the constraints, and SCIP search then brings them closer to the original point.
The attack is evaluated on four datasets (LCLD, URL, WIDS, CTU) and compared with [Moeva2](https://github.com/serval-uni-lu/constrained-attacks).

This repository is the implementation of the following paper (SFPVV2026):

> Takumi Kubo, Takahisa Toda and Hisashi Koga. *Exploiting Discrete Structures for Constrained Adversarial Attacks on Tree Ensembles*. SFPVV2026.

## Table of Contents
- [Environment Setup](#environment-setup)
- [Usage](#usage)
- [Development](#development)
- [Contents of this repository](#contents-of-this-repository)
- [License](#license)

## Environment Setup
We recommend using [uv](https://docs.astral.sh/uv/).
1. If you don't have uv, please prepare it following [Installation | uv](https://docs.astral.sh/uv/getting-started/installation/). (It should be as simple as running `curl -LsSf https://astral.sh/uv/install.sh | sh`)
2. Run `uv sync`.

With the above steps, you will have an environment that works with Python 3.11.

If you use venv, you will have a hard time making it work with 3.11. This is because [Moeva2](https://github.com/serval-uni-lu/constrained-attacks) requires `python = "~3.8"` (although it actually works with 3.11). We used uv during the experiments to run it on 3.11, and while writing the documentation, we realized that setting up the environment with venv is difficult.

## Usage
1. Extract the data used in the experiments (models and target points for attack) by running `tar -xvf dataset.tar.xz`.
2. Run `main.py`. For uv, use `uv run main.py`, and for venv, use `python3 main.py`. If you run it as is, you should get the data written in the paper.

## Development
Development tools (pytest, ruff, pyrefly) are installed by `uv sync`.
```bash
uv run pytest         # unit tests (tests/)
uv run ruff check     # lint
uv run ruff format    # format
uv run pyrefly check  # type check
```
Run the tests from the repository root, because they refer to `tests/sample_xgboost.json` by a relative path.

## Contents of this repository
| Path | Description |
|---|---|
| [`main.py`](#mainpy) | Entry point that runs the experiments |
| [`smt_attack/`](#smt_attack) | Source code of the attack method |
| [`constraints/`](#constraints) | Constraints for each dataset |
| [`tests/`](#tests) | Unit tests |
| [`dataset.tar.xz`](#datasettarxz) | Models and target points for the attack |
| [`raw_results/`](#raw_results) | Experimental results referred to in the paper |
| [`tables/`](#tables) | Aggregated tables output by `utils/collect_*.py` |
| [`original_data/`](#original_data) | Distributed raw data and metadata |
| [`utils/`](#utils) | Scripts not directly related to the method |

### `main.py`
This is the entry point. To change the execution parameters (like the model), please edit the following `cases`.
```py
cases = [
    Case(
        "dataset/lcld/xgboost_100.json",
        "dataset/lcld/x_test.npy",
        range(100),
        LcldCostraints,
        "output",
    ),
]
```
The way to write a Case is as follows. Please also refer to `main.py`.
```py
@dataclass
class Case:
    model_path: str  # Path to the model
    # Path to the target points for the attack
    # The target points are located at `dataset/{name}/x_test.npy` in the directory extracted from dataset(dataset.tar.xz)
    # Example: for url, it would be `dataset/url/x_test.npy`
    test_data_path: str
    # Specify the target points for the attack. range(10) means the first 10 points in the test data will be targeted.
    test_range: range
    # Specify the constraint object
    # For lcld, import LcldCostraints with `from constraints.lcld import LcldCostraints` and pass it
    # For url, import UrlCostraints with `from constraints.url import UrlCostraints` and pass it
    # For wids, import WidsCostraints with `from constraints.wids import WidsCostraints` and pass it
    # For ctu, import CtuCostraints with `from constraints.ctu import CtuCostraints` and pass it
    constraints: Callable[[Point], BaseConstraints]
    # Specify the directory to write the results to. If not specified, the files will be created in the current directory.
    dump_dir: str | None = None
```

### `smt_attack`
Source code for the attack method.
- `smt_search.py`: The part that runs the SMT solver to find initial points.
- `scip_search.py`: Brings the points closer using SCIP search based on the initial points.

### `dataset.tar.xz`
A compressed file containing the target points for attack and the models trained with XGBoost dumped. Since `main.py` refers to this, it must be extracted before execution.

The directory after extraction looks like this:
```
.
|-- ctu
|   |-- x_test.npy
|   |-- x_train.npy
|   |-- xgboost_10.json
|   |-- xgboost_100.json
|   |-- xgboost_200.json
|   |-- xgboost_50.json
|   |-- y_test.npy
|   `-- y_train.npy
|-- lcld
|   `-- (same as ctu)
|-- url
|   `-- (same as ctu)
`-- wids
    `-- (same as ctu)
```
Each directory corresponds to its respective dataset. `x_test` and `x_train` are the test data and training data, and `y_test` and `y_train` are the correct labels for the test data and training data. `xgboost_{k}.json` is the decision tree ensemble trained on each dataset consisting of k decision trees.

### `constraints`
Constraints for each dataset.

### `tests`
Unit tests run with pytest. `tests/sample_xgboost.json` is a small tree used as test data, and `tests/dummy_constraints.py` is a minimal `BaseConstraints` implementation shared by the tests.

### `raw_results`
Experimental results referred to in the paper. `baseline` contains existing comparison results, and `with_t_bound` contains comparison results with and without `t_bound`.

### `tables`
Contains the aggregated tabular data outputted by `utils/collect_*.py`.

### `original_data`
A compressed file containing the distributed raw data and metadata (representing type and border constraints written by Simonetto). Each directory corresponds to a dataset.

The raw data is in CSV format. The metadata is also in CSV format and is written in the following format:
```csv
feature,type,mutable,min,max
length_url,int,True,12.0,1641.0
length_hostname,int,True,4.0,214.0
```

### `utils`
Source code not directly related to the method.
- `collect_baseline.py`, `collect_data_t_bound.py`: Reads the data output by `main.py` and outputs the tables referred to in the paper to the `tables` directory.
- `generate_tree.py`: For training trees, etc.

## License
See [LICENSE](LICENSE).
