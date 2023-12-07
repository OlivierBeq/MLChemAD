# -*- coding: utf-8 -*-

"""Toy data."""


import os
import json

import pandas as pd


def _read_data():
    # Obtain a file handle
    with open(os.path.abspath(os.path.join(__file__, os.pardir, 'data.json'))) as fh:
        toy_data = json.load(fh)
    return (toy_data.get('source'),
            pd.DataFrame(toy_data.get('training')).set_index('Compound'),
            pd.DataFrame(toy_data.get('test')).set_index('Compound')
            )


source, training, test = _read_data()
