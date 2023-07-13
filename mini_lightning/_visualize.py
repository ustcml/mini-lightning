# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:
from ._types import *

__all__ = ['tensorboard_smoothing', 'read_tensorboard_file', 'plot_image']
Item = Dict[str, float]  # e.g. keys of Item: step, value
COLOR, COLOR_S = '#FFE2D9', '#FF7043'


def read_tensorboard_file(fpath: str) -> Dict[str, List[Item]]:
    """
    return: keys of dict: e.g. 'train_loss'...
    """
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f'fpath: {fpath}')
    ea = EventAccumulator(fpath)
    ea.Reload()
    res = {}
    tags = ea.Tags()['scalars']
    for tag in tags:
        values = ea.Scalars(tag)
        r = []
        for v in values:
            r.append({'step': v.step, 'value': v.value})
        res[tag] = r
    return res


def tensorboard_smoothing(values: List[float], smooth: float = 0.9) -> List[float]:
    """
    values: List[`value` of Item]. You don't need to pass in `step`
    """
    # [0.81 0.9 1]. res[2] = (0.81 * values[0] + 0.9 * values[1] + values[2]) / 2.71
    norm_factor = 1
    x = 0
    res = []
    for i in range(len(values)):
        if math.isnan(values[i]):
            res.append(float('nan'))
            continue
        v = values[i]
        x = x * smooth + v  # Exponential decay
        res.append(x / norm_factor)
        #
        norm_factor *= smooth
        norm_factor += 1
    return res


def plot_image(tb_dir: str, smooth_key: List[str], smooth_val: float = 0.9,
               figsize: Tuple[int, int] = (8, 5), dpi: int = 100) -> None:
    image_dir = os.path.join(os.path.dirname(tb_dir), 'images')
    os.makedirs(image_dir, exist_ok=True)
    #
    fname = os.listdir(tb_dir)[0]
    tb_path = os.path.join(tb_dir, fname)
    data = read_tensorboard_file(tb_path)
    #
    for k in data.keys():
        _data = data[k]
        steps = [d['step'] for d in _data]
        values = [d['value'] for d in _data]
        if len(values) == 0:
            continue
        _, ax = plt.subplots(1, 1, squeeze=True, figsize=figsize, dpi=dpi)
        ax.set_title(k)
        if len(values) == 1:
            ax.scatter(steps, values, color=COLOR_S)
        elif k in smooth_key:
            ax.plot(steps, values, color=COLOR)
            values_s = tensorboard_smoothing(values, smooth_val)
            ax.plot(steps, values_s, color=COLOR_S)
        else:
            ax.plot(steps, values, color=COLOR_S)
        fpath = os.path.join(image_dir, k)
        plt.savefig(fpath, dpi=dpi, bbox_inches='tight')
