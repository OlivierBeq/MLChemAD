# -*- coding: utf-8 -*-

"""Toy data."""


import json
import os
from typing import List

import pandas as pd


class Dataset:
    """Collection of data for training and testing."""

    def __init__(self, name: str, path: str, in_memory: bool = False):
        """Create a dataset from a name and file path.

        The file the path points to should follow the template that examples follow.

        :param name: name of the dataset
        :param path: path to a file containing the data of the dataset defined by the 'source', 'training' and 'test' sections.
        :param in_memory: should properties be kept in memory once one has been accessed to.
        Otherwise, read from file upon each property access.
        """
        if not os.path.isfile(path):
            raise ValueError('file does not exist')
        self.name = name
        self.path = path
        self.in_memory = in_memory

    @property
    def source(self):
        if hasattr(self, '_source'):
            return getattr(self, '_source')
        else:
            toy_data = self._load()
            return toy_data['source']

    @property
    def training(self):
        if hasattr(self, '_training'):
            return getattr(self, '_training')
        else:
            toy_data = self._load()['training']
            return pd.DataFrame(toy_data['data'],
                                index=toy_data['index'],
                                columns=toy_data['columns'])

    @property
    def test(self):
        if hasattr(self, '_test'):
            return getattr(self, '_test')
        else:
            toy_data = self._load()['test']
            return self.to_pandas(toy_data)

    def _load(self):
        with open(self.path) as fh:
            toy_data = json.load(fh)
        if self.in_memory:
            self._source = toy_data['source']
            self._training = self.to_pandas(toy_data['training'])
            self._test = self.to_pandas(toy_data['test'])
        return toy_data

    @staticmethod
    def to_pandas(data):
        """Convert json-decoded data into a pandas dataframe."""
        return pd.DataFrame(data['data'],
                            index=data['index'],
                            columns=data['columns'])

    def __repr__(self):
        return f'<Dataset: (training shape: {self.training.shape}; test shape {self.test.shape})>'


class Datasets:
    """Collection of datasets."""

    def __init__(self, data: List[Dataset] = None):
        """Create a collection of datasets.

        :param data: list of datasets
        """
        if data is None:
            self._datasets = {}
        else:
            if len(set(dataset.name for dataset in data)) != len(data):
                raise ValueError('all dataset names must be unique')
            self._datasets = {dataset.name: dataset
                              for dataset in data}

    def __getitem__(self, item):
        """Obtain a dataset from its name."""
        return self._datasets.get(item)

    def __getattr__(self, item):
        return self._datasets.get(item)

    @property
    def names(self):
        return tuple(self._datasets.keys())


def _read_data(in_memory: bool = False):
    # Obtain path to folder
    path = os.path.dirname(__file__)
    # Obtain files in the folder
    data_files = [f for f in os.listdir(path) if f.endswith('.json')]
    # Obtain datasets
    datasets = [Dataset(os.path.splitext(filepath)[0],
                        os.path.abspath(os.path.join(path, filepath)),
                        in_memory=in_memory)
                for filepath in data_files]
    return Datasets(datasets)


data = _read_data()
