# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from ._types import *

__all__ = ['tensorboard_smoothing', 'read_tensorboard_file']
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
        x = x * smooth + values[i]  # Exponential decay
        res.append(x / norm_factor)
        #
        norm_factor *= smooth
        norm_factor += 1
    return res


def plot_image(data: Dict[str, List[Item]], key_name: str, smooth: float,
               figsize: Tuple[int, int] = (8, 5), dpi: int = 100) -> Figure:
    _data = data[key_name]
    steps = [d['step'] for d in _data]
    values = [d['value'] for d in _data]
    fig, ax = plt.subplots(1, 1, squeeze=True, figsize=figsize, dpi=dpi)
    ax.set_title(key_name)
    if smooth != 0:
        ax.plot(steps, values, color=COLOR)
        values_s = tensorboard_smoothing(values, smooth)
        ax.plot(steps, values_s, color=COLOR_S)
    else:
        ax.plot(steps, values, color=COLOR_S)
    return fig
