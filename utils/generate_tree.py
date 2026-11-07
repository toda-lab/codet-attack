from constraints.ctu import CtuCostraints
import logging as _logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from constraints.wids import WidsCostraints

import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split


logger = _logging.getLogger(__name__)


class Data(NamedTuple):
    data: NDArray[np.float64]
    target: NDArray[np.float64 | np.int64]


@dataclass
class DataSetuper:
    name: str
    load: Callable[[], Data]


def load_lcld() -> Data:
    """Load the LCLD dataset.

    Returns:
        Loaded data.

    """
    df = pd.read_csv('dataset/lcld_v2/lcld_v2.csv')

    # Convert issue_d from a date to an integer number of days since 1970-01-01
    if 'issue_d' in df.columns:
        # pyrefly: ignore [no-matching-overload]
        df['issue_d'] = pd.to_datetime(df['issue_d']).astype('period[D]').astype('int64')

    # Split into output (charged_off) and input (all other columns) and convert to numpy arrays
    target = df['charged_off'].to_numpy()
    data = df.drop(columns=['charged_off']).to_numpy()

    return Data(data=data, target=target)


def load_wids() -> Data:
    """Load the WiDS dataset, removing rows that violate the constraints.

    Returns:
        Loaded data.

    """
    df = pd.read_csv('dataset/wids/wids.csv')
    raw_xs = df.iloc[:, 1:]
    raw_ys = df.iloc[:, 0]

    x_data = []
    y_data = []
    # Remove rows that violate the constraints
    for raw_x, raw_y in zip(raw_xs.iterrows(), list(raw_ys), strict=True):
        x = np.array(raw_x[1])
        y = np.array(raw_y)
        if not WidsCostraints(x).is_valid(x):
            continue
        x_data.append(x)
        y_data.append(y)

    return Data(data=np.array(x_data), target=np.array(y_data))


def load_url() -> Data:
    """Load the URL dataset.

    Returns:
        Loaded data.

    """
    df = pd.read_csv('dataset/url/url.csv')
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1].map({'legitimate': 0, 'phishing': 1})
    return Data(data=x.to_numpy(), target=y.to_numpy())


def load_ctu() -> Data:
    """Load the CTU dataset, removing rows that violate the constraints.

    Returns:
        Loaded data.

    """
    df = pd.read_csv('dataset/ctu/ctu.csv')
    raw_xs = df.iloc[:, :-1]
    raw_ys = df.iloc[:, -1]

    x_data = []
    y_data = []
    # Remove rows that violate the constraints
    idx = 0
    for raw_x, raw_y in zip(raw_xs.iterrows(), list(raw_ys), strict=True):
        x = np.array(raw_x[1])
        y = np.array(raw_y)
        constraints = CtuCostraints(x)
        if not constraints.is_valid(x):
            continue
        x_data.append(x)
        y_data.append(y)
        idx += 1
        if idx % 10 == 0:
            logger.debug(f'{idx = }')

    return Data(data=np.array(x_data), target=np.array(y_data))


lcld = DataSetuper(
    name='lcld',
    load=load_lcld,
)

wids = DataSetuper(
    name='wids',
    load=load_wids,
)

url = DataSetuper(
    name='url',
    load=load_url,
)

ctu = DataSetuper(
    name='ctu',
    load=load_ctu,
)

# XGBoost training parameters
PARAM = {
    'objective': 'binary:logistic',
    'eta': 0.3,
    'gamma': 1.0,
    'min_child_weight': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'colsample_bynode': 0.8,
    'num_parallel_tree': -1,
    'max_depth': 6,
    'tree_method': 'hist',
}


def main() -> None:
    """Train XGBoost models for each dataset and save them in the LT-Attack format.

    Returns:
        None.

    """
    _logging.basicConfig(level=_logging.DEBUG)
    setuppers = [lcld]
    # note: load_cts() takes some hours.
    # setuppers = [lcld, wids, url, ctu]
    num_trees = [10]

    output = Path('local_data')
    output.mkdir(exist_ok=True)

    for setupper in setuppers:
        name = setupper.name
        logger.info(f'loading {name}...')
        directory = output / name
        directory.mkdir(exist_ok=True)
        data = setupper.load()
        X = data.data
        y = data.target

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        np.save(directory / 'x_train', X_train)
        np.save(directory / 'x_test', X_test)
        np.save(directory / 'y_train', y_test)
        np.save(directory / 'y_test', y_test)

        dtrain = xgb.DMatrix(X_train, label=y_train)

        for num_tree in num_trees:
            logger.info(f'learning model with {num_tree} trees')
            param = PARAM.copy()
            param['num_parallel_tree'] = num_tree

            bst = xgb.train(param, dtrain, num_boost_round=1)

            # Save the model as JSON and parse it into the format used by LT-Attack

            model_json_path = directory / f'xgboost_{num_tree}.json'
            bst.dump_model(model_json_path, dump_format='json')


if __name__ == '__main__':
    main()
