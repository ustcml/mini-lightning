import unittest as ut
#
import torch
import torch.nn as nn
from torch import optim
#
import mini_lightning as ml


class TestUtils(ut.TestCase):
    def test_de_sync_batchnorm(self):
        # inplace
        m = nn.Sequential(
            nn.Linear(10, 10),
            nn.BatchNorm2d(10)
        )
        m2 = nn.SyncBatchNorm.convert_sync_batchnorm(m)
        self.assertTrue(m == m2)
        self.assertTrue(m[1] == m2[1])
        m3 = ml.de_sync_batchnorm(m, "2d")
        self.assertTrue(m == m2 == m3)
        self.assertTrue(m[1] == m2[1] == m3[1])
        del m, m2, m3
        # not inplace
        m = nn.BatchNorm2d(10)
        m2 = nn.SyncBatchNorm.convert_sync_batchnorm(m)
        self.assertTrue(m != m2)
        m3 = ml.de_sync_batchnorm(m, "2d")
        self.assertTrue(m != m2 and m2 != m3)
        self.assertTrue(m == m3)

    def test_utils(self):
        # test seed_everything
        s = ml.seed_everything(3234335211)
        print(s)
        # test time_synchronize
        x = torch.randn(10000, 10000, device='cuda')
        # test test_time
        res = ml.test_time(lambda: x @ x, 10, 0, ml.time_synchronize)

    def test_print_model_info(self):
        # test print_model_info
        from torchvision.models import resnet50
        import torch
        model = resnet50()
        input = torch.randn(1, 3, 224, 224)
        ml.print_model_info("resnet", model, (input, ))
        ml.print_model_info("resnet", model)
        ml.print_model_info("resnet", model, (input, ))

    def test_ckpt(self):
        model = nn.Linear(10, 10)
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        for i in range(10):
            x = torch.randn(10, 10)
            loss = model(x).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        ml.save_ckpt("asset/tmp.ckpt", {"model": model}, [optimizer], 0)
        #
        models, optimizer_state_dict, mes = ml.load_ckpt("asset/tmp.ckpt")
        optimizer2 = optim.SGD(models["model"].parameters(), lr=0.1, momentum=0.9)
        optimizer2.load_state_dict(optimizer_state_dict[0])
        print(mes)


if __name__ == "__main__":
    ut.main()
