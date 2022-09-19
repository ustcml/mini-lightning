# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from pre import *
import gym
from gym import Env
#

"""
Training and exploration are decoupled.
1. Each iteration will be trained once. The dataset used for training is randomly sampled from the memory pool.
2. At the same time, the Agent also makes an exploration. 
    Make decision and take an action (random or model decision). 
    Change from state to next_state. And then put it in the memory pool.
It will warm up memory pool at first. Fill in some memory.
"""


RENDER = True
RUNS_DIR = os.path.join(RUNS_DIR, "dqn")
DATASETS_PATH = os.environ.get("DATASETS_PATH", os.path.join(RUNS_DIR, "datasets"))
CHECKPOINTS_PATH = os.path.join(RUNS_DIR, "checkpoints")
os.makedirs(DATASETS_PATH, exist_ok=True)
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)

#
device_ids = [0]


class DQN(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


Memory = namedtuple(
    "Memory",
    # ndarray, int, float, bool, ndarray
    ["state", "action", "reward", "done", "next_state"]
)


class MemoryPool:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.pool: Deque[Memory] = deque(maxlen=capacity)

    def len(self) -> int:
        return len(self.pool)

    def __getitem__(self, key) -> Memory:
        return self.pool[key]

    def add(self, memo: Memory) -> None:
        self.pool.append(memo)

    def sample(self) -> Memory:
        idx = np.random.choice(len(self.pool), (), replace=False)
        return self[idx]


class MyDataset(IterableDataset):
    def __init__(self, memo_pool: MemoryPool, dataset_len: int) -> None:
        self.memo_pool = memo_pool
        self.dataset_len = dataset_len

    def __iter__(self) -> Iterator:
        for _ in range(self.dataset_len):
            yield self.memo_pool.sample()

    def __len__(self) -> int:
        return self.dataset_len


class Agent:
    def __init__(self, env: Env, memo_pool: MemoryPool, model: Module, device: Union[str, Device]) -> None:
        self.env = env
        self.memo_pool = memo_pool
        self.model = model
        self.device = Device(device) if isinstance(device, str) else device
        #
        self.state = None
        self.reset_env()
        if RENDER:
            self.env.render()

    def reset_env(self) -> None:
        self.state = self.env.reset()

    def step(self, rand_p: float) -> Tuple[float, bool]:
        action = self._get_action(rand_p)
        next_state, reward, done, _ = self.env.step(action)
        memo = Memory(self.state, action, reward, done, next_state)
        self.memo_pool.add(memo)
        if done:
            self.reset_env()
        else:
            self.state = next_state
        if RENDER:
            self.env.render()
        return reward, done  # for log

    def _get_action(self, rand_p: float) -> int:
        if np.random.random() < rand_p:
            return self.env.action_space.sample()
        #
        state = torch.from_numpy(self.state)[None].to(self.device)
        q_value: Tensor = self.model(state)[0]
        return int(q_value.argmax(dim=0).item())


def get_rand_p(global_step: int, T_max: int, eta_min: float, eta_max: float) -> float:
    rand_p = ml.cosine_annealing_lr(global_step, T_max, eta_min, [eta_max])[0]
    return rand_p


class MyLModule(ml.LModule):
    def __init__(self, model: Module, optim: Optimizer, loss_fn: Module, agent: Agent, get_rand_p: Callable[[int], float],
                 hparams: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(model, optim, {}, None, hparams)
        self.old_model = deepcopy(model)
        self.old_model.eval()
        # New_model and old_model are used for model training.
        # New model is used for exploring the environment and training.
        # Old model is used for calculating the reward prediction of next_state.
        # Reason: to eliminate associations.
        self.loss_fn = loss_fn
        self.agent = agent
        self.get_rand_p = get_rand_p
        #
        self.warmup_memory_steps = self.hparams["warmup_memory_steps"]
        # synchronize the model every sync_steps
        self.sync_steps = self.hparams["sync_steps"]
        self.gamma = self.hparams["gamma"]  # reward decay

        #
        self._warmup_memo(self.warmup_memory_steps)
        self.episode_reward = 0  # reward in a episode

    def trainer_init(self, trainer: "ml.Trainer") -> None:
        super().trainer_init(trainer)
        self.old_model.to(trainer.device)

    def _warmup_memo(self, steps: int) -> None:
        for _ in tqdm(range(steps), desc=f"Warmup: "):
            self.agent.step(rand_p=1)

    def _train_step(self, batch: Any) -> Tensor:
        states, actions, rewards, dones, next_states = batch
        q_values = self.model(states)
        # The Q value of taking the action in the current state (Scalar) is approximated with
        # reward+gamma*(Q value of taking the best action in next_state (0 if done=True))
        y_pred = q_values[torch.arange(len(actions)), actions]
        with torch.no_grad():
            y_true: Tensor = self.old_model(next_states).max(1)[0]
            y_true[dones] = 0.
        y_true.mul_(self.gamma).add_(rewards)
        loss = self.loss_fn(y_pred, y_true)
        return loss

    @torch.no_grad()
    def _agent_step(self) -> Tuple[float, bool]:
        rand_p = self.get_rand_p(self.global_step)
        reward, done = self.agent.step(rand_p)
        if done:
            self.episode_reward = 0
        else:
            self.episode_reward += reward
        return reward, done

    def training_step(self, batch: Any) -> Tensor:
        if self.global_step % self.sync_steps == 0:
            # copy state dict
            self.old_model.load_state_dict(self.model.state_dict())
        # train model
        loss = self._train_step(batch)
        # agent step in env
        reward, done = self._agent_step()
        # log
        self.log("reward", reward, prog_bar_mean=False)
        self.log("done", done, prog_bar_mean=False)
        self.log("episode_reward", self.episode_reward)
        self.log("loss", loss)
        return loss


if __name__ == "__main__":
    ml.seed_everything(42, gpu_dtm=False)
    batch_size = 32
    max_epochs = 20
    hparams = {
        "device_ids": device_ids,
        "memo_capacity": 1000,
        "dataset_len": 5000,
        "env_name": "CartPole-v1",
        "model_hidden_size": 128,
        "optim_name": "SGD",
        "dataloader_hparams": {"batch_size": batch_size},
        "optim_hparams": {"lr": 1e-2, "weight_decay": 1e-4, "momentum": 0.9},  #
        "trainer_hparams": {"max_epochs": max_epochs, "gradient_clip_norm": 50},
        #
        "rand_p": {
            "eta_max": 1,
            "eta_min": 0,
            "T_max": ...
        },
        "sync_steps": 20,  # synchronization frequency of old_model
        "warmup_memory_steps": 1000,  # warm up and fill the memory pool
        "gamma": 0.99,  # reward decay

    }
    memo_pool = MemoryPool(hparams["memo_capacity"])
    dataset = MyDataset(memo_pool, hparams["dataset_len"])
    ldm = ml.LDataModule(
        dataset, None, None, **hparams["dataloader_hparams"], shuffle_train=False, num_workers=1)
    hparams["rand_p"]["T_max"] = ml.get_T_max(len(dataset), batch_size, max_epochs, 1)
    env = gym.make(hparams["env_name"])
    in_channels: int = env.observation_space.shape[0]
    out_channels: int = env.action_space.n
    model = DQN(in_channels, out_channels, hparams["model_hidden_size"])
    agent = Agent(env, memo_pool, model, ml.select_device(device_ids))

    #
    get_rand_p = partial(get_rand_p, **hparams["rand_p"])
    optimizer = getattr(optim, hparams["optim_name"])(model.parameters(), **hparams["optim_hparams"])
    runs_dir = CHECKPOINTS_PATH
    loss_fn = nn.MSELoss()

    lmodel = MyLModule(model, optimizer, loss_fn, agent, get_rand_p, hparams)
    trainer = ml.Trainer(lmodel, device_ids, runs_dir=runs_dir, **hparams["trainer_hparams"])
    trainer.fit(ldm.train_dataloader, ldm.val_dataloader)
