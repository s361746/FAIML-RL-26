
import numpy as np
import gymnasium as gym

class RandomizationWrapper(gym.Wrapper):
    """
    Wrapper that applies randomization to the environment
    """

    SUPPORTED_MODES = {"none", "udr", "adr"}
    DOMAIN_MASS_RANGES = {
        "source": (1.0, 1.0),
        "target": (1.5, 2.5),
    }

    def __init__(
        self,
        env,
        env_type="source",
        mode="none",
        mass_range=None,
    ):
        super().__init__(env)

        self.env_type = env_type
        self.mode = mode

        if mass_range is None:
            self.mass_min_limit, self.mass_max_limit = self.DOMAIN_MASS_RANGES[env_type]
        else:
            self.mass_min_limit, self.mass_max_limit = mass_range

        self.mass_min = self.mass_min_limit
        self.mass_max = self.mass_max_limit
        self.current_mass = None
        self.last_sample_type = "fixed"

        self.episode_reward = 0.0
        self.episode_count = 0
        self.reward_history = []
        self.adr_adjustment = 0.03


    def _sample_mass(self):
        if self.mode == "none":
            self.last_sample_type = "fixed"
            if self.mass_min_limit == self.mass_max_limit:
                return float(self.mass_min_limit)
            return float((self.mass_min_limit + self.mass_max_limit) / 2.0)

        if self.mode == "udr":
            self.last_sample_type = "uniform"
            return float(np.random.uniform(self.mass_min_limit, self.mass_max_limit))

        if self.mode == "adr":
            self.last_sample_type = "adaptive"
            return float(np.random.uniform(self.mass_min, self.mass_max))

        raise ValueError(f"Unsupported sampling mode '{self.mode}'.")

    def _update_adr_bounds(self):
        if self.mode != "adr" or len(self.reward_history) < 2:
            return

        previous_reward = self.reward_history[-2]
        latest_reward = self.reward_history[-1]
        span = self.mass_max_limit - self.mass_min_limit
        adjustment = max(1e-4, self.adr_adjustment * span)

        if latest_reward >= previous_reward:
            self.mass_min = max(self.mass_min_limit, self.mass_min - adjustment)
            self.mass_max = min(self.mass_max_limit, self.mass_max + adjustment)
            self.last_sample_type = "expanded"
        else:
            self.mass_min = min(self.mass_max, self.mass_min + adjustment)
            self.mass_max = max(self.mass_min, self.mass_max - adjustment)
            self.last_sample_type = "contracted"

        self.mass_min = float(np.clip(self.mass_min, self.mass_min_limit, self.mass_max_limit))
        self.mass_max = float(np.clip(self.mass_max, self.mass_min, self.mass_max_limit))


    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.episode_reward += float(reward)
        done = terminated or truncated

        if done:
            self.episode_count += 1
            self.reward_history.append(self.episode_reward)
            self._update_adr_bounds()
            self.episode_reward = 0.0

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        new_mass = self._sample_mass()

        if new_mass is not None:
            self.current_mass = new_mass
            sim = self.env.unwrapped.task.sim
            object_body_id = sim._bodies_idx["object"]

            sim.physics_client.changeDynamics(
                bodyUniqueId=object_body_id,
                linkIndex=-1,
                mass=float(new_mass),
            )

            print(
                f"[{self.env_type}:{self.mode}] mass={new_mass:.2f} "
                f"range=[{self.mass_min:.2f},{self.mass_max:.2f}] "
                f"type={self.last_sample_type}"
            )

        observation, info = super().reset(**kwargs)
        info = dict(info)
        info["randomized_mass"] = self.current_mass
        return observation, info
