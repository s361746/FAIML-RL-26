import numpy as np
import gymnasium as gym

class RandomizationWrapper(gym.Wrapper):
    """
    Wrapper that applies randomization to the environment
    """

    SUPPORTED_MODES = {"none", "udr", "adr"}
    DOMAIN_MASS_RANGES = {
        "source": (1.0, 9.0),
        "target": (5.0, 5.0),
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

        # Start from tight bounds for ADR, expand dynamically
        if self.mode == "adr":
            self.mass_min = 1.0
            self.mass_max = 1.0
        else:
            self.mass_min = self.mass_min_limit
            self.mass_max = self.mass_max_limit

        self.current_mass = None
        self.last_sample_type = "fixed"

        self.episode_successes = []
        self.window_size = 20         
        self.adr_adjustment = 0.05
        
        self.thr_high = 0.80           
        self.thr_low = 0.40           

    def _sample_mass(self):
        if self.mode == "none":
            self.last_sample_type = "fixed"
            return float(self.mass_min_limit)  # Baseline source is 1.0kg, target eval is 5.0kg

        if self.mode == "udr":
            self.last_sample_type = "uniform"
            return float(np.random.uniform(self.mass_min_limit, self.mass_max_limit))

        if self.mode == "adr":
            self.last_sample_type = "adaptive"
            return float(np.random.uniform(self.mass_min, self.mass_max))

        raise ValueError(f"Unsupported sampling mode '{self.mode}'.")

    def _update_adr_bounds(self, is_success):
        if self.mode != "adr":
            return

        self.episode_successes.append(is_success)
        
        if len(self.episode_successes) >= self.window_size:
            avg_success = np.mean(self.episode_successes)
            self.episode_successes.clear()

            span = self.mass_max_limit - self.mass_min_limit
            adjustment = max(1e-4, self.adr_adjustment * span)

            if avg_success >= self.thr_high:
                self.mass_min = max(self.mass_min_limit, self.mass_min - adjustment)
                self.mass_max = min(self.mass_max_limit, self.mass_max + adjustment)
                self.last_sample_type = "expanded"
            elif avg_success < self.thr_low:
                self.mass_min = min(self.mass_max, self.mass_min + adjustment)
                self.mass_max = max(self.mass_min, self.mass_max - adjustment)
                self.last_sample_type = "contracted"
            else:
                self.last_sample_type = "maintained"

            self.mass_min = float(np.clip(self.mass_min, self.mass_min_limit, self.mass_max_limit))
            self.mass_max = float(np.clip(self.mass_max, self.mass_min, self.mass_max_limit))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        if done and self.mode == "adr":
            is_success = float(info.get("is_success", 0.0))
            self._update_adr_bounds(is_success)

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