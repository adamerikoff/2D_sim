import random
import time
import math

import pygame

from constants import WIDTH, HEIGHT, WIND_FORCE_MAX, PIXELS_PER_METER, RENDER_PAUSE

from drone import Drone
from grenade import Grenade
from target import Target
from vector import Vector

class Environment:
    def __init__(self, dt=0.1, drone_min_height = 0.5, screen_width=200, screen_height=190, train=True):
        self.dt = dt
        self.drone_min_height = drone_min_height
        self.width = screen_width
        self.height = screen_height
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width*PIXELS_PER_METER, screen_height*PIXELS_PER_METER))
        self.font = pygame.font.SysFont(None, 20)
        self.train = train

    def reset(self):
        self.drone = Drone(random.randint(0, WIDTH), random.randint(0, HEIGHT-(HEIGHT*self.drone_min_height)))
        self.grenade = self.drone.attach_grenade()
        self.target = Target(random.randint(40, self.width-40), self.height)
        self.wind = self._generateWindForce()
        self.steps = 0
        self.score = 0
        self.prev_distance = None

        return self._get_observation()

    def step(self, action):
        if self.grenade.released and self.train:
            reward = 0
            while not self._is_done():
                self.drone.update(action, self.dt)
                self.grenade.update(self.wind, self.dt)
                self.steps += 1
                reward += self._calculate_reward()
            done = self._is_done()
            observation = self._get_observation()
            return observation, reward, done, {}
        else:
            self.steps += 1
            self.drone.update(action, self.dt)
            self.grenade.update(self.wind, self.dt)
            done = self._is_done()
            reward = self._calculate_reward()
            observation = self._get_observation()
            return observation, reward, done, {}

    def _get_observation(self):
        distance_to_target = Vector.euclidean_distance(self.target.coordinates, self.drone.coordinates)

        new_p1 = self.target.coordinates.x - self.drone.coordinates.x
        new_p2 = self.target.coordinates.y - self.drone.coordinates.y

        angle = math.degrees(math.atan2(new_p1, new_p2))

        observation = [
            self.drone.coordinates.x, self.drone.coordinates.y,
            distance_to_target,
            angle,
            self.wind.x, self.wind.y,
        ]
        return observation

    def _calculate_reward(self):
        distance = Vector.euclidean_distance(self.grenade.coordinates, self.target.coordinates)
        if self.grenade.hit_ground:
            if distance < 5:
                return 100
            else:
                return -distance
        if self.grenade.released and self.prev_distance is not None:
            if self.prev_distance > distance:
                return 1
        self.prev_distance = distance
        return -1
    
    def _generateWindForce(self):
        return Vector(random.uniform(-WIND_FORCE_MAX, WIND_FORCE_MAX), 0)
    
    def _is_done(self):
        return self.grenade.hit_ground

    def render(self):
        self.screen.fill((180, 180, 180))  # Clear the screen
        # Draw objects (dummy example)
        self.drone.render(self.screen, PIXELS_PER_METER)
        self.grenade.render(self.screen, PIXELS_PER_METER)
        self.target.render(self.screen, PIXELS_PER_METER)
        self._draw_info()
        pygame.display.flip()
        time.sleep(RENDER_PAUSE)

    def close(self):
        if self.renderMode:
            pygame.quit()

    def _draw_scale(self):
        for y in range(0, HEIGHT, 20):
            text = self.font.render(f"{y} meters", True, (0, 0, 0))
            self.screen.blit(text, (5, (self.height - y) * PIXELS_PER_METER))

    def _draw_time_counter(self):
        total_time = self.steps * self.dt  # Total simulation time
        minutes = int(total_time // 60)  # Integer division to get seconds
        seconds = total_time % 60  # Get the remainder for microseconds
        rounded_seconds = round(seconds, 2)  # Round to two decimal places for seconds

        time_text = f"Time: {minutes:02}:{rounded_seconds:05.2f}"  # Format as mm:ss.xx
        text = self.font.render(time_text, True, (0, 0, 0))
        self.screen.blit(text, (self.width * PIXELS_PER_METER - 150, 10))

    def _draw_grenade_velocity(self):
        velocity_magnitude = self.grenade.velocity.magnitude()
        terminal_velocity_text = f"T.Velocity: {self.grenade.terminal_velocity:.2f} m/s"
        velocity_text = f"Velocity: {velocity_magnitude:.2f} m/s"

        text = self.font.render(velocity_text, True, (0, 0, 0))
        self.screen.blit(text, (self.width * PIXELS_PER_METER - 150, 30))
        text = self.font.render(terminal_velocity_text, True, (0, 0, 0))
        self.screen.blit(text, (self.width * PIXELS_PER_METER - 150, 50))

    def _draw_debug(self):
        observation = self._get_observation()
        # Observation values to display
        x, y, distance_to_target, angle, wind_x, wind_y = observation
        # Convert angle to degrees for display
        angle_degrees = angle
        # Create text surfaces
        x_text = self.font.render(f"Drone X: {x:.2f} m", True, (0, 0, 0))
        y_text = self.font.render(f"Drone Y: {y:.2f} m", True, (0, 0, 0))
        distance_text = self.font.render(f"Distance to Target: {distance_to_target:.2f} m", True, (0, 0, 0))
        angle_text = self.font.render(f"Angle: {angle_degrees:.2f} degrees", True, (0, 0, 0))
        wind_x_text = self.font.render(f"Wind X: {wind_x:.2f} m/s", True, (0, 0, 0))
        wind_y_text = self.font.render(f"Wind Y: {wind_y:.2f} m/s", True, (0, 0, 0))
        # Blit the text onto the screen
        self.screen.blit(x_text, (self.width * PIXELS_PER_METER - 150, 70))
        self.screen.blit(y_text, (self.width * PIXELS_PER_METER - 150, 90))
        self.screen.blit(distance_text, (self.width * PIXELS_PER_METER - 150, 110))
        self.screen.blit(angle_text, (self.width * PIXELS_PER_METER - 150, 130))
        self.screen.blit(wind_x_text, (self.width * PIXELS_PER_METER - 150, 150))
        self.screen.blit(wind_y_text, (self.width * PIXELS_PER_METER - 150, 170))


    def _draw_info(self):
        self._draw_scale()
        self._draw_time_counter()
        self._draw_grenade_velocity()
        self._draw_debug()


    

