import pygame

from vector import Vector

class Target:
    def __init__(self, x, y, width=10, height=10):
        self.coordinates = Vector(x, y)  # Position of the drone (top-left corner)
        self.width = width  # Width of the drone in meters
        self.height = height  # Height of the drone in meters

    def render(self, screen, pixel_per_meter):
        # Convert world coordinates and dimensions to screen coordinates
        screen_x = int((self.coordinates.x - self.width / 2) * pixel_per_meter)
        screen_y = int((self.coordinates.y - self.height / 2) * pixel_per_meter)
        screen_width = int(self.width * pixel_per_meter)
        screen_height = int(self.height * pixel_per_meter)

        # Draw the drone as a rectangle
        pygame.draw.rect(screen, (100, 150, 50), (screen_x, screen_y, screen_width, screen_height))