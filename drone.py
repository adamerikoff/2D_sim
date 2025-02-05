import pygame

from constants import WIDTH, HEIGHT
from vector import Vector
from grenade import Grenade

class Drone:
    def __init__(self, x, y, width=5, height=2):
        self.coordinates = Vector(x, y)  # Position of the drone (top-left corner)
        self.width = width  # Width of the drone in meters
        self.height = height  # Height of the drone in meters
        self.grenade = None

    def update(self, action, dt):
        if action == 0:
            self.coordinates.x += 10 * dt  # Move right
            if not self.grenade.released:
                self.grenade.coordinates.x = self.coordinates.x
                self.grenade.coordinates.y = self.coordinates.y + 1
        elif action == 1:
            self.coordinates.x -= 10 * dt  # Move left
            if not self.grenade.released:
                self.grenade.coordinates.x = self.coordinates.x
                self.grenade.coordinates.y = self.coordinates.y + 1
        elif action == 2:
            if not self.grenade.released:
                self.grenade.released = True

    def attach_grenade(self):
        self.grenade = Grenade(self.coordinates.x, self.coordinates.y + 1, HEIGHT)
        return self.grenade

    def render(self, screen, pixel_per_meter):
        # Convert world coordinates and dimensions to screen coordinates
        screen_x = int((self.coordinates.x - self.width / 2) * pixel_per_meter)
        screen_y = int((self.coordinates.y - self.height / 2) * pixel_per_meter)
        screen_width = int(self.width * pixel_per_meter)
        screen_height = int(self.height * pixel_per_meter)

        # Draw the drone as a rectangle
        pygame.draw.rect(screen, (0, 0, 255), (screen_x, screen_y, screen_width, screen_height))