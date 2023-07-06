# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from pre import *
#
from PIL import Image
import torchvision.transforms as tvt
import torchvision.datasets as tvd
import torchvision.models as tvm
from torchvision.utils import make_grid, save_image
#
MNIST = tvd.MNIST
CIFAR10 = tvd.CIFAR10
CIFAR100 = tvd.CIFAR100
STL10 = tvd.STL10
ResNet = tvm.ResNet
DenseNet = tvm.DenseNet



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
    save_image(imgs[idxs].flatten(0, 1), fpath, nrow=topk, padding=2, value_range=(-1, 1), pad_value=1)
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
    plt.savefig(tsne_fpath, bbox_inches="tight")
    plt.close()
    logger.info(f"`draw_tsne` Done. The image is saved in `{tsne_fpath}`")

# for meta-learning


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
    """copy from `torchvision.models.densenet`"""
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


# for contrastive_learning

def NT_Xent_loss(features: Tensor, temperature: float = 0.1) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    features: FloatTensor. [2N, E]
    return: loss: [], pos_idx_mean: [2N], acc: [2N], acc_5: [2N]
    """
    NINF = -torch.inf
    device = features.device
    N = features.shape[0] // 2
    cos_sim = pairwise_cosine_similarity(features, features)
    cos_sim = cos_sim / temperature
    self_mask = torch.arange(2 * N, dtype=torch.long, device=device)
    pos_mask = self_mask.roll(N)  # [2N]
    cos_sim[self_mask, self_mask] = NINF
    pos_sim = cos_sim[self_mask, pos_mask]
    #
    loss = -pos_sim + torch.logsumexp(cos_sim, dim=-1)
    loss = loss.mean()
    #
    pos_sim = pos_sim.clone().detach_()[:, None]  # [2N, 1]
    cos_sim = cos_sim.clone().detach_()  # [2N, 2N]
    cos_sim[self_mask, pos_mask] = NINF  # ignore
    comb_sim = torch.concat([pos_sim, cos_sim], dim=-1)
    # idx of pos_sim
    pos_idx = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
    acc = (pos_idx == 0).float()
    acc_5 = (pos_idx < 5).float()
    return loss, pos_idx, acc, acc_5


class GatherLayer(Function):
    """ref: https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/gather.py"""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tuple[Tensor]:
        res = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(res, x)
        return tuple(res)

    @staticmethod
    def backward(ctx, *grads: Tensor) -> Tensor:
        res = grads[ml.get_dist_setting()[0]]
        res *= dist.get_world_size()  # for same grad with W * batch_size; mean operation in ddp across device.
        return res


def pairwise_euclidean_distance(
    X: Tensor,
    Y: Tensor,
    squared: bool = False
) -> Tensor:
    """
    X: shape[N1, F]. FloatTensor
    Y: shape[N2, F]. FloatTensor
    return: shape[N1, N2]
    """
    XX = torch.einsum("ij,ij->i", X, X)
    YY = torch.einsum("ij,ij->i", Y, Y)
    # 
    res = X @ Y.T
    res.mul_(-2).add_(XX[:, None]).add_(YY)
    res.clamp_min_(0.)
    return res if squared else res.sqrt_()