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
from torchvision.models._api import WeightsEnum
#
MNIST = tvd.MNIST
CIFAR10 = tvd.CIFAR10
CIFAR100 = tvd.CIFAR100
STL10 = tvd.STL10
ResNet = tvm.ResNet
DenseNet = tvm.DenseNet


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

#


@torch.no_grad()
def prepare_features(model: Module, dataset: Dataset, device: Device) -> TensorDataset:
    """
    3 step: to; eval; no_grad
    """
    loader: DataLoader = ml.LDataModule(None, dataset, None, 64, 4).val_dataloader
    model.eval()
    model.to(device)
    features, labels = [], []
    for x, y in tqdm(loader, desc="Prepare Features"):
        x, y = ml.LModule.batch_to_device((x, y), device=device)
        f: Tensor = model(x)
        features.append(f)
        labels.append(y)
    #
    features = torch.concat(features, dim=0)
    labels = torch.concat(labels, dim=0)
    #
    return TensorDataset(features.cpu(), labels.cpu())


def draw_similar_images(dataset: TensorDataset,
                        imgs_np: ndarray, topk: int, fpath: str) -> None:
    """
    dataset: x: [N, F] float. y: [N] long
    imgs_np: [N, C, H, W]
    """
    x, y = dataset.tensors
    imgs = torch.from_numpy(imgs_np).div(255)
    y, idxs = y.sort()
    x, imgs = x[idxs], imgs[idxs]
    qx = [x[y == i][0] for i in range(10)]
    qx = torch.stack(qx, dim=0)
    #
    cos_sim = pairwise_cosine_similarity(qx, x)
    idxs = cos_sim.topk(topk, dim=1)[1]
    #
    save_images(imgs[idxs].flatten(0, 1), topk, fpath, norm=True)
    logger.info(f"`draw_similar_images` Done. The image is saved in `{fpath}`")


def draw_tsne(dataset: TensorDataset, tsne_fpath: str, TSNE: type) -> None:
    """
    dataset: x: [N, F] float. y: [N] long
    """
    x, y = dataset.tensors
    tsne = TSNE(2, learning_rate="auto", init="random", n_jobs=4)
    x_2d = tsne.fit_transform(x.numpy())
    for label in range(10):
        plt.scatter(x_2d[:, 0][y == label], x_2d[:, 1][y == label], s=20, alpha=0.5, label=label)
    plt.legend()
    plt.savefig(tsne_fpath, bbox_inches='tight')
    plt.close()
    logger.info(f"`draw_tsne` Done. The image is saved in `{tsne_fpath}`")


class ImageDataset(Dataset):
    def __init__(
        self,
        images: ndarray,
        targets: Tensor,
        transform: Optional[Callable[[Any], Any]] = None
    ) -> None:
        super().__init__()
        self.images = images
        self.targets = targets
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        image, target = self.images[idx], self.targets[idx]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self) -> int:
        return self.images.shape[0]


def load_densenet_state_dict(
    model: nn.Module,
    state_dict: Dict[str, Any],
    strict: bool = False
) -> IncompatibleKeys:
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    return model.load_state_dict(state_dict, strict=strict)
