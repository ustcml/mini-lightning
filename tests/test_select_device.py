
import mini_lightning as ml
import unittest as ut
import torch


class TestSelectDevice(ut.TestCase):
    def test_select_device(self):

        ml.select_device([0])
        x = torch.randn((10,))
        ml.select_device([0])
        x = torch.randn((10,), device="cuda:0")
        ml.select_device([0])


"""
[INFO: mini-lightning] Using device: cuda:0
[INFO: mini-lightning] Using device: cuda:0
[WARNING: mini-lightning] CUDA has been initialized! Device selection fails!
"""

if __name__ == "__main__":
    ut.main()
