"""
LeRobot data source.
"""

from typing import List, Optional
import torch

from ..base import DataSource, TrajectoryData


class LeRobotOccupancyDataSource(DataSource):
    """
    Data source for LeRobot datasets from HuggingFace.
    """

    def __init__(
        self,
        repo_id: str,
        episodes: Optional[List[int]] = None,
        n_rollout: Optional[int | float] = None,
        preload_trajectories: bool = False,
        root: Optional[str] = None,
        image_key: str = "observation.images.image",
        pad_action_dim: Optional[int] = None,
        explicit_episodes: bool = False,
        num_tasks: int = 50,
    ):
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
            from torchvision.transforms import v2
        except ImportError:
            raise ImportError(
                "LeRobot is required for LeRobotDataSource. "
                "Install with: pip install lerobot"
            )

        self.repo_id = repo_id
        self.root = root
        self.image_key = image_key
        self.v2 = v2
        self.num_tasks = num_tasks
        self._task_name_to_id: dict[str, int] = {}
        self._warned_task_fallback = False
        # Zero-pad the action's trailing dim to this size at load time. Used by
        # sim→real finetune to match a pretrained ActionEmbedder trained on a
        # larger effective action dim (e.g. digital PushT 2D × frame_interval=5).
        self.pad_action_dim = pad_action_dim

        if episodes is None and n_rollout is not None:
            if isinstance(n_rollout, float) and 0 < n_rollout < 1:
                temp_dataset = LeRobotDataset(
                    repo_id=repo_id,
                    root=root,
                    image_transforms=None,
                    episodes=None
                )
                num_total = temp_dataset.num_episodes
                n_episodes = int(num_total * n_rollout)
                episodes = list(range(n_episodes))
                print(f"Loading {n_rollout*100:.1f}% of episodes: {n_episodes}/{num_total}")
            else:
                episodes = list(range(int(n_rollout)))
                print(f"Loading first {n_rollout} episodes")

        if episodes is None and explicit_episodes:
            metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
            episodes = list(range(metadata.total_episodes))
            print(f"Loading all {metadata.total_episodes} episodes explicitly")

        print(f"Loading LeRobot dataset: {repo_id} (root={root})")
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            image_transforms=None,
            episodes=episodes
        )

        self.num_episodes = self.dataset.num_episodes
        self.preload_trajectories = preload_trajectories

        self.stats = {
            "action_mean": self.dataset.meta.stats["action"]["mean"],
            "action_std": self.dataset.meta.stats["action"]["std"],
            "state_mean": self.dataset.meta.stats["observation.state"]["mean"],
            "state_std": self.dataset.meta.stats["observation.state"]["std"],
        }

        self.trajectories = None
        self.trajectory_cache: dict[int, TrajectoryData] = {}
        if self.preload_trajectories:
            print("Preloading trajectory data into memory...")
            self._preload_trajectories()
            # _load_single_trajectory already applies padding.
            self._action_dim = self.trajectories[0].actions.shape[-1]
            self._state_dim = self.trajectories[0].states.shape[-1]
        else:
            native_action_dim, self._state_dim = self._infer_dims()
            if self.pad_action_dim is not None:
                if self.pad_action_dim < native_action_dim:
                    raise ValueError(
                        f"pad_action_dim ({self.pad_action_dim}) must be >= "
                        f"native action_dim ({native_action_dim})"
                    )
                self._action_dim = self.pad_action_dim
            else:
                self._action_dim = native_action_dim

        # Pad stats so downstream normalization (if any) sees a consistent shape.
        # mean=0, std=1 for padded dims keeps normalized padding at 0.
        # lerobot stores stats as numpy arrays in some versions; handle both.
        if self.pad_action_dim is not None:
            import numpy as np
            def _pad_stat(arr, fill_value):
                cur = int(arr.shape[0])
                if self.pad_action_dim < cur:
                    raise ValueError(
                        f"pad_action_dim ({self.pad_action_dim}) must be >= "
                        f"native action_dim ({cur})"
                    )
                if self.pad_action_dim == cur:
                    return arr
                extra = self.pad_action_dim - cur
                if isinstance(arr, torch.Tensor):
                    fill = torch.full((extra,), fill_value, dtype=arr.dtype)
                    return torch.cat([arr, fill])
                fill = np.full((extra,), fill_value, dtype=arr.dtype)
                return np.concatenate([arr, fill])

            self.stats["action_mean"] = _pad_stat(self.stats["action_mean"], 0.0)
            self.stats["action_std"] = _pad_stat(self.stats["action_std"], 1.0)

        print(f"Loaded {self.num_episodes} episodes from {repo_id}")
        print(f"  State dim: {self._state_dim}, Action dim: {self._action_dim}")

    def _infer_dims(self) -> tuple[int, int]:
        # Sample a few episodes and assert dim consistency — catches heterogeneous
        # datasets (different cameras/robots) rather than silently using ep 0's dims.
        num_eps = self.num_episodes
        probe_indices = sorted(set([0, num_eps // 2, num_eps - 1])) if num_eps > 0 else [0]

        action_dims, state_dims = set(), set()
        for ep_idx in probe_indices:
            start, _ = self._episode_global_range(ep_idx)
            item = self.dataset[start]
            state_dims.add(int(item["observation.state"].shape[-1]))
            action_dims.add(int(item["action"].shape[-1]))

        if len(action_dims) > 1 or len(state_dims) > 1:
            raise ValueError(
                f"LeRobot dataset has heterogeneous dims across episodes "
                f"(action_dims={action_dims}, state_dims={state_dims}). "
                "Expected a single-robot dataset."
            )
        return action_dims.pop(), state_dims.pop()

    def _preload_trajectories(self) -> None:
        self.trajectories = []

        for ep_idx in range(self.num_episodes):
            self.trajectories.append(self._load_single_trajectory(ep_idx))

    def _episode_global_range(self, index: int) -> tuple[int, int]:
        """Resolve (global_start, global_end) frame indices for an episode.

        lerobot v2.x stores `episode_data_index["from"/"to"]` as int tensors
        mapping episode_idx -> global frame offsets. The `episode_index` field
        of meta.episodes is the episode *number*, not a frame offset — using it
        as `start_idx` was a bug that only worked by coincidence for ep 0.
        """
        idx_map = self.dataset.episode_data_index
        start = int(idx_map["from"][index])
        end = int(idx_map["to"][index])
        return start, end

    def _coerce_task_id(self, value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return int(value.item())
            value = value.detach().cpu().tolist()
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            if value not in self._task_name_to_id:
                self._task_name_to_id[value] = len(self._task_name_to_id)
            return self._task_name_to_id[value]
        if isinstance(value, dict):
            for key in ("task_index", "task_id", "task", "name"):
                task_id = self._coerce_task_id(value.get(key))
                if task_id is not None:
                    return task_id
        return None

    def _task_from_episode_metadata(self, index: int):
        episodes = getattr(getattr(self.dataset, "meta", None), "episodes", None)
        if episodes is None:
            return None
        try:
            if hasattr(episodes, "iloc"):
                episode = episodes.iloc[index]
            elif isinstance(episodes, dict):
                episode = episodes.get(index, episodes.get(str(index)))
            else:
                episode = episodes[index]
        except Exception:
            return None
        return self._coerce_task_id(episode)

    def _task_from_item(self, item):
        for key in ("task_index", "task_id", "task", "task_name"):
            if key in item:
                task_id = self._coerce_task_id(item[key])
                if task_id is not None:
                    return task_id
        return None

    def _task_id_for_episode(self, index: int, first_item=None) -> int:
        task_id = self._task_from_item(first_item) if first_item is not None else None
        if task_id is None:
            task_id = self._task_from_episode_metadata(index)
        if task_id is None:
            raise ValueError(
                "[LeRobotOccupancyDataSource] Could not find task metadata for "
                f"episode {index} in {self.repo_id}. Occupancy training requires "
                "a valid task id/name from the item or episode metadata; refusing "
                "to infer task from episode_index."
            )
        if not (0 <= int(task_id) < self.num_tasks):
            raise ValueError(
                f"Task id {task_id} for episode {index} is outside "
                f"[0, {self.num_tasks})."
            )
        return int(task_id)

    def _load_single_trajectory(self, index: int) -> TrajectoryData:
        start_idx, end_idx = self._episode_global_range(index)
        length = end_idx - start_idx

        states = []
        actions = []
        first_item = None

        for frame_idx in range(start_idx, end_idx):
            item = self.dataset[frame_idx]
            if first_item is None:
                first_item = item
            states.append(item["observation.state"])
            actions.append(item["action"])

        states_tensor = torch.stack(states, dim=0)
        actions_tensor = torch.stack(actions, dim=0)
        if self.pad_action_dim is not None:
            cur = actions_tensor.shape[-1]
            if self.pad_action_dim > cur:
                pad = torch.zeros(
                    actions_tensor.shape[0],
                    self.pad_action_dim - cur,
                    dtype=actions_tensor.dtype,
                )
                actions_tensor = torch.cat([actions_tensor, pad], dim=-1)

        return TrajectoryData(
            states=states_tensor,
            actions=actions_tensor,
            seq_length=length,
            meta={
                "episode_index": index,
                "global_start": start_idx,
                "task_index": self._task_id_for_episode(index, first_item),
            }
        )

    def load_trajectory(self, index: int) -> TrajectoryData:
        if index >= self.num_episodes:
            raise IndexError(f"Index {index} out of range [0, {self.num_episodes})")
        if self.preload_trajectories:
            return self.trajectories[index]
        if index in self.trajectory_cache:
            return self.trajectory_cache[index]
        traj = self._load_single_trajectory(index)
        self.trajectory_cache[index] = traj
        return traj

    def load_visual_frames(
        self,
        index: int,
        start: int,
        end: int,
        step: int = 1
    ) -> torch.Tensor:
        abs_start, _ = self._episode_global_range(index)

        frames = []
        for rel_idx in range(start, end, step):
            abs_idx = abs_start + rel_idx
            item = self.dataset[abs_idx]
            raw = item[self.image_key]
            # lerobot returns float [3,H,W] in [0,1] for decoded video frames.
            # to_image() only handles uint8 / PIL inputs, so branch on dtype.
            if isinstance(raw, torch.Tensor) and raw.dtype.is_floating_point:
                frame = raw
            else:
                frame = self.v2.functional.to_image(raw)
            frames.append(frame)

        frames_tensor = torch.stack(frames, dim=0)
        if frames_tensor.dtype == torch.uint8:
            frames_tensor = frames_tensor.float() / 255.0

        return frames_tensor

    def get_num_trajectories(self) -> int:
        return self.num_episodes

    def get_seq_length(self, index: int) -> int:
        # Avoid loading the entire trajectory just to report length —
        # lerobot exposes per-episode lengths in metadata.
        start, end = self._episode_global_range(index)
        return int(end - start)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def state_dim(self) -> int:
        return self._state_dim
