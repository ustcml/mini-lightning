import unittest as ut
#
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
#
import os
import mini_lightning as ml
from torchvision.models import resnet18


class TestUtils(ut.TestCase):
    def test_de_sync_batchnorm(self) -> None:
        # inplace
        m = nn.Sequential(
            nn.Linear(10, 10),
            nn.BatchNorm2d(10)
        )
        m2 = nn.SyncBatchNorm.convert_sync_batchnorm(m)
        self.assertTrue(m == m2)
        self.assertTrue(m[1] == m2[1])
        m3 = ml.de_sync_batchnorm(m, '2d')
        self.assertTrue(m == m2 == m3)
        self.assertTrue(m[1] == m2[1] == m3[1])
        del m, m2, m3
        # not inplace
        m = nn.BatchNorm2d(10)
        m2 = nn.SyncBatchNorm.convert_sync_batchnorm(m)
        self.assertTrue(m != m2)
        m3 = ml.de_sync_batchnorm(m, '2d')
        self.assertTrue(m != m2 and m2 != m3)
        self.assertTrue(m == m3)

    def test_utils(self) -> None:
        # test seed_everything
        s = ml.seed_everything(3234335211)
        print(s)
        # test time_synchronize
        x = torch.randn(10000, 10000, device='cuda')
        # test test_time
        res = ml.test_time(lambda: x @ x, 10, 0, ml.time_synchronize)

    def test_print_model_info(self) -> None:
        # test print_model_info
        from torchvision.models import resnet50
        import torch
        model = resnet50()
        input = torch.randn(1, 3, 224, 224)
        ml.print_model_info(model, None, (input, ))
        ml.print_model_info(model, 'resnet', )
        ml.print_model_info(model, 'resnet', (input, ))

    def test_ckpt(self) -> None:
        model = resnet18()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        def lr_lambda(epoch: int) -> float:
            # 返回scale
            if epoch < 10:
                return 1.
            elif epoch < 20:
                return 0.1
            else:
                return 0.01
        #
        lr_s: _LRScheduler = LambdaLR(optimizer, lr_lambda)
        for _ in range(10):
            x = torch.randn(16, 3, 224, 224)
            loss = model(x).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        os.makedirs('asset', exist_ok=True)
        ml.save_ckpt('asset/tmp.ckpt', {'model': model.state_dict()}, [], [])
        ml.save_ckpt('asset/tmp2.ckpt', {'model': model.state_dict()}, [optimizer], [lr_s])
        #
        models_state_dict, _, _,  mes = ml.load_ckpt('asset/tmp.ckpt')
        model = models_state_dict['model']
        print(mes)


if __name__ == '__main__':
    ut.main()
