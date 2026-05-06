from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class DeepSeaScavenger(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.window_size = 768
        self.x, self.y = 0.0, 0.0
        self.torque, self.angle, self.speed = 0.0, 0.0, 0.0
        self.oxygen = 1.0
        self.is_sonar_active = 0.0
        self.sonar_readings = np.zeros(12, dtype=np.float32)
        self.treasure_pos = np.zeros(2, dtype=np.float32)
        self.floor_heights = None

        """
        Observation Space:
            - Submarine coordinates (x and y)
            - Torque
            - Angle
            - Speed
            - Oxygen level
            - Sonar activity
            - 12 sonar readings
        19 features overall
        """
        self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, 
                    shape=(19,), 
                    dtype=np.float32
                )

        """
        Action Space
            - Engine
            - Torque
            - Active sonar
        """
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0]), 
            high=np.array([1.0, 1.0, 1.0]), 
            dtype=np.float32
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
            If human-rendering is used, `self.window` will be a reference
            to the window that we draw to. `self.clock` will be a clock that is used
            to ensure that the environment is rendered at the correct framerate in
            human-mode. They will remain `None` until human-mode is used for the
            first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        state = np.array([
            self.x, self.y, self.torque, self.angle, 
            self.speed, self.oxygen, float(self.is_sonar_active)
        ], dtype=np.float32)
        
        return np.concatenate([state, self.sonar_readings]).astype(np.float32)

    def _get_info(self):
            return {
                "distance_to_treasure": np.linalg.norm(
                    np.array([self.x, self.y]) - self.treasure_pos
                ),
                "oxygen_left": self.oxygen
            }

    def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            
            self.x = self.np_random.uniform(50.0, 100.0)
            self.y = self.np_random.uniform(50.0, 100.0)
            
            self.torque = 0.0
            self.angle = 0.0
            self.speed = 0.0
            
            self.oxygen = 1.0
            self.is_sonar_active = 0.0
            self.sonar_readings = np.zeros(12, dtype=np.float32)

            num_points = 12
            x_points = np.linspace(0, self.window_size, num_points)
            y_points = 600 + self.np_random.uniform(-150, 100, num_points)
            x_all = np.arange(self.window_size)
            base_terrain = np.interp(x_all, x_points, y_points)

            sine_waves = (
                np.sin(x_all * 0.02) * 20 + 
                np.sin(x_all * 0.05) * 10
            )

            noise = self.np_random.normal(0, 3, self.window_size)

            self.floor_heights = base_terrain + sine_waves + noise
        
            self.floor_heights = np.clip(self.floor_heights, 100, self.window_size - 50)

            self.treasure_pos = np.array([
                self.np_random.uniform(100.0, 500.0),
                self.np_random.uniform(100.0, 500.0)
            ], dtype=np.float32)

            observation = self._get_obs()
            info = self._get_info()

            if self.render_mode == "human":
                self._render_frame()

            return observation, info
    
    def _check_collision(self):
        return False
    
    def _cast_rays(self):
        return np.zeros(12, dtype=np.float32)

    def step(self, action):
            # Reading action
            engine = action[0]
            torque = action[1]
            sonar_req = action[2]

            # Physics
            self.angle += torque * 0.1
            self.speed += engine * 0.05
            self.speed = np.clip(self.speed, -0.5, 1.0)
            
            # Updating the position of the submarine
            self.x += np.cos(self.angle) * self.speed
            self.y += np.sin(self.angle) * self.speed

            # Sonar and oxygen
            self.is_sonar_active = 1.0 if sonar_req > 0.5 else 0.0
            self.oxygen -= 0.001
            if self.is_sonar_active:
                self.oxygen -= 0.005
                self.sonar_readings = self._cast_rays() # TBI

            # Check for termination condition (hit the bottom or ran out of oxygen)
            terminated = self.oxygen <= 0 or self._check_collision()
            
            # Reward
            dist = np.linalg.norm(np.array([self.x, self.y]) - self.treasure_pos)
            reward = -0.01
            if dist < 20.0: reward = 10.0

            # 6. Preparing info
            observation = self._get_obs()
            info = self._get_info()

            if self.render_mode == "human":
                self._render_frame()

            return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
            if self.window is None and self.render_mode == "human":
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
            if self.clock is None and self.render_mode == "human":
                self.clock = pygame.time.Clock()

            canvas = pygame.Surface((self.window_size, self.window_size))
            canvas.fill((20, 20, 50))

            # Displaying the bottom
            points = []
            for x in range(self.window_size):
                points.append((x, self.floor_heights[x]))
            
            pygame.draw.lines(canvas, (100, 100, 100), False, points, 3)

            # Displaying treasure chest
            pygame.draw.circle(canvas, (255, 215, 0), self.treasure_pos.astype(int), 10)

            # Displaying submarine
            agent_pos = np.array([self.x, self.y]).astype(int)
            pygame.draw.circle(canvas, (0, 255, 255), agent_pos, 15)
            
            end_pos = agent_pos + np.array([np.cos(self.angle), np.sin(self.angle)]) * 25
            pygame.draw.line(canvas, (255, 255, 255), agent_pos, end_pos.astype(int), 3)

            # Rendering
            if self.render_mode == "human":
                self.window.blit(canvas, canvas.get_rect())
                pygame.event.pump()
                pygame.display.update()
                self.clock.tick(self.metadata["render_fps"])
            else:
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
                )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()



env = DeepSeaScavenger(render_mode="human")
obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample() 
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated:
        obs, info = env.reset()

env.close()