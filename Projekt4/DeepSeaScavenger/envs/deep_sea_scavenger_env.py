from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class DeepSeaScavenger(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.window_size = 768
        self.x, self.y = 100.0, 150.0
        self.ship_width = 40.0
        self.ship_height = 20.0
        self.torque, self.angle, self.speed = 0.0, 0.0, 0.0
        self.font = None
        self.max_oxygen = 100.0
        self.oxygen = 100.0
        self.is_sonar_active = 0.0
        self.sonar_readings = np.zeros(12, dtype=np.float32)
        self.treasure_pos = []
        self.treasures_collected = 0
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
            
            self.x = 100.0
            self.y = 150.0
            self.torque = 0.0
            self.angle = 0.0
            self.speed = 0.0
            
            self.oxygen = 100.0
            self.max_oxygen = 100.0
            self.is_sonar_active = 0.0
            self.sonar_readings = np.zeros(12, dtype=np.float32)

            self.treasure_pos = []
            self.treasures_collected = 0

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


            for _ in range(3):
                treasure_x = self.np_random.integers(50, self.window_size - 50)
                treasure_y = self.floor_heights[int(treasure_x)] - 15
                self.treasure_pos.append((treasure_x, treasure_y))

            observation = self._get_obs()
            info = self._get_info()

            if self.render_mode == "human":
                self._render_frame()

            return observation, info
    
    def _check_collision(self):
        hw = self.ship_width / 2
        hh = self.ship_height / 2
        
        vertices = [
            (-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)
        ]
        
        cos_a = np.cos(self.angle)
        sin_a = np.sin(self.angle)
        
        for vx, vy in vertices:
            rx = vx * cos_a - vy * sin_a
            ry = vx * sin_a + vy * cos_a
            
            world_x = int(self.x + rx)
            world_y = self.y + ry
            
            if 0 <= world_x < self.window_size:
                if world_y >= self.floor_heights[world_x]:
                    return True
            else:
                 return True
        return False
    
    def _cast_rays(self, dist):
        num_rays = 12
        steps = 10 
        
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        dists = np.linspace(0, dist, steps)
        
        dx = np.sin(angles)[:, np.newaxis]
        dy = np.cos(angles)[:, np.newaxis]
        
        test_x = (self.x + dx * dists).astype(int)
        test_y = (self.y + dy * dists)
        
        test_x = np.clip(test_x, 0, self.window_size - 1)
        hit = test_y >= self.floor_heights[test_x]
        
        readings = np.ones(num_rays, dtype=np.float32)
        for i in range(num_rays):
            hits = np.where(hit[i])[0]
            if hits.size > 0:
                readings[i] = dists[hits[0]] / dist
                
        return readings

    def step(self, action):
            # Reading action
            engine = action[0]
            torque = action[1]
            sonar_req = action[2]

            # Physics
            self.angle += torque * 0.2
            self.speed += engine * 0.3
            self.speed = np.clip(self.speed, -5, 10.0)
            
            # Updating the position of the submarine
            self.x += np.cos(self.angle) * self.speed
            self.y += np.sin(self.angle) * self.speed

            # Block swimming over the surface and regenerate oxygen
            surface_limit = 100 + (self.ship_height / 2)
            if self.y < surface_limit:
                self.y = surface_limit
                self.vy = 0

            if self.y - (self.ship_height / 2) <= 101:
                self.oxygen += 8 
                if self.oxygen > self.max_oxygen:
                    self.oxygen = self.max_oxygen

            # Sonar and oxygen
            self.is_sonar_active = 1.0 if sonar_req > 0.5 else 0.0
            self.oxygen -= 0.2
            if self.is_sonar_active:
                self.oxygen -= 3
                self.sonar_readings = self._cast_rays(dist=150)
            else:
                self.sonar_readings = self._cast_rays(dist=50)


            # Check for termination condition (hit the bottom or ran out of oxygen)
            terminated = self.oxygen <= 0 or self._check_collision()
            
            # Reward
            distance_to_treasure = np.linalg.norm(np.array([self.x, self.y]) - self.treasure_pos)
            reward = -0.1

            if terminated:
                if self.oxygen <= 0:
                    reward = -20.0
                else:
                    reward = -50.0

            # Collecting treasuers
            for i in range(len(self.treasure_pos) - 1, -1, -1):
                curr_treasure_pos = self.treasure_pos[i]
                dist = np.linalg.norm(np.array([self.x, self.y]) - curr_treasure_pos)
                
                if dist < 20.0:
                    reward += 100.0
                    self.treasures_collected += 1
                    self.oxygen = min(self.max_oxygen, self.oxygen + 20)
                    self.treasure_pos.pop(i)

            # Preparing info
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

            # Background (ocean)
            canvas = pygame.Surface((self.window_size, self.window_size))
            canvas.fill((20, 20, 50))

            # Surface
            surface_rect = pygame.Rect(0, 0, self.window_size, 100)
            pygame.draw.rect(canvas, (0, 150, 255), surface_rect)

            # Displaying the bottom
            points = []
            for x in range(self.window_size):
                points.append((x, self.floor_heights[x]))
            
            pygame.draw.lines(canvas, (100, 100, 100), False, points, 3)

            # Displaying treasure chest
            for treasure in self.treasure_pos:
                pygame.draw.circle(canvas, (255, 215, 0), treasure, 10)

            # Displaying submarine
            surface = pygame.Surface((self.ship_width, self.ship_height), pygame.SRCALPHA)
            surface.fill((0, 50, 150))
            
            rotated_surf = pygame.transform.rotate(surface, -np.degrees(self.angle))
            rect = rotated_surf.get_rect(center=(int(self.x), int(self.y)))
            canvas.blit(rotated_surf, rect.topleft)

            # Sonar rays
            num_rays = 12
            dist = 150.0 if self.is_sonar_active > 0.5 else 30.0
            angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
            
            for i, angle in enumerate(angles):
                dx = np.sin(angle)
                dy = np.cos(angle)
                
                dist = self.sonar_readings[i] * dist
                
                color = (255, 0, 0) if self.sonar_readings[i] < 1.0 else (0, 255, 0)
                
                start_pos = (int(self.x), int(self.y))
                end_pos = (int(self.x + dx * dist), int(self.y + dy * dist))
                pygame.draw.line(canvas, color, start_pos, end_pos, 2)

            # Oxygen level
            if self.font is None:
                pygame.font.init()
                self.font = pygame.font.SysFont("Arial", 24, bold=True)

            oxygen_text = f"Oxygen: {int(self.oxygen)} / {int(self.max_oxygen)}"
            
            if self.oxygen > 50:
                text_color = (100, 255, 100)
            elif self.oxygen > 20:
                text_color = (255, 255, 0)
            else:
                text_color = (255, 50, 50)
                
            text_surface = self.font.render(oxygen_text, True, text_color)
            canvas.blit(text_surface, (20, 20))

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

running = True
while running:
    keys = pygame.key.get_pressed()
    
    action = [0.0, 0.0, 0.0]
    
    if keys[pygame.K_UP]:    action[0] = 1.0
    if keys[pygame.K_DOWN]:  action[0] = -1.0
    if keys[pygame.K_LEFT]:  action[1] = -1.0
    if keys[pygame.K_RIGHT]: action[1] = 1.0
    if keys[pygame.K_SPACE]: action[2] = 1.0
    
    obs, reward, terminated, truncated, info = env.step(np.array(action, dtype=np.float32))
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
    if terminated or truncated:
        obs, info = env.reset()

env.close()
