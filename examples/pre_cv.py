# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from pre import *
#
from PIL import Image
import torchvision.transforms as tvt
import torchvision.datasets as tvd
import torchvision.models as tvm
from torchvision.utils import make_grid
#
MNIST = tvd.MNIST
CIFAR10 = tvd.CIFAR10
STL10 = tvd.STL10
ResNet = tvm.ResNet


def save_images(
    images: Tensor, ncols: int, path: str, *,
    norm: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    pad_value: float = 0.
) -> None:
    """copy from gan.py
    images: [N, C, H, W]
    """
    images = images.detach().cpu()
    N = images.shape[0]
    nrows = int(math.ceil(N / ncols))
    images = make_grid(images, nrow=ncols, normalize=norm, value_range=value_range,
                       pad_value=pad_value)  # [C, H, W], 0-1
    images.clip_(0, 1)
    images = images.permute(1, 2, 0).numpy()
    #
    fig, ax = plt.subplots(figsize=(2 * ncols, 2 * nrows), dpi=200)
    ax.imshow(images, cmap=None, origin="upper", vmin=0, vmax=1)
    ax.axis("off")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
