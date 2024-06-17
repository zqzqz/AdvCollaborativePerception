import os
import pickle
import numpy as np

from test_base import *


def test_opv2v():
    from mvp.data.opv2v_dataset import OPV2VDataset
    dataset = OPV2VDataset(root_path=os.path.join(root, "data/OPV2V"), mode="test")
    case = dataset.get_case(0, tag="multi_vehicle")
    if len(case) > 0:
        print("Dataset is available")


if __name__ == "__main__":
    test_opv2v()