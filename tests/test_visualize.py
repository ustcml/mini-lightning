import os
import unittest as ut
#
import matplotlib.pyplot as plt
import mini_lightning as ml

class TestVisualize(ut.TestCase):
    def test_tensorboard_utils(self) -> None:
        # run in mini-lightning folder
        fpath = "./asset/events.out.tfevents.1658302059.jintao.13896.0"
        loss = ml.read_tensorboard_file(fpath)["train_loss"]
        v = [l["value"] for l in loss]
        step = [l["step"] for l in loss]
        sv = ml.tensorboard_smoothing(v, 0.9)  # smoothing_v
        print(sv[490//5 - 1], v[490//5-1])

        def plot_loss() -> None:
            _, ax = plt.subplots(figsize=(10, 5))
            cg, cb = "#FFE2D9", "#FF7043"
            ax.plot(step, v, color=cg)  # color grey
            ax.plot(step, sv, color=cb)  # color bright
        plot_loss()
        os.makedirs("./asset/images", exist_ok=True)
        plt.savefig("./asset/images/1.png", dpi=200, bbox_inches="tight")
        # plt.show()


if __name__ == "__main__":
    ut.main()
