import tkinter as tk
import random
import math

class Player:
    def __init__(self, canvas, size=30):
        self.canvas = canvas
        self.size = size
        self.image = canvas.create_oval(0, 7, size, size, fill='yellow')
        self.pollen_circles = {}  # Dictionary to store pollen circles by color
        self.x_speed = 0
        self.y_speed = 0
        self.pollen_inventory = {'red': 0, 'blue': 0, 'purple': 0, 'orange': 0, 'pink': 0}
        self.pollen_labels = {}
        self.update_pollen_labels()

    def move_towards(self, x, y, speed=5):
        # Move the player towards the specified coordinates
        x_diff = x - self.get_x()
        y_diff = y - self.get_y()
        distance = math.sqrt(x_diff * x_diff + y_diff * y_diff)

        if distance != 0:
            # Calculate the normalized direction vector
            x_normalized = x_diff / distance
            y_normalized = y_diff / distance

            # Update the player's speed based on the direction
            self.x_speed = x_normalized * speed
            self.y_speed = y_normalized * speed
        else:
            self.x_speed = 0
            self.y_speed = 0

    def move(self):
        # Move the player based on its speed
        self.canvas.move(self.image, self.x_speed, self.y_speed)

        # Create or update the pollen circles based on the inventory
        for index, (color, count) in enumerate(self.pollen_inventory.items()):
            circle = self.pollen_circles.get(color)

            # Delete existing circle if count is 0
            if circle and count == 0:
                self.canvas.delete(circle)
                self.pollen_circles[color] = None
            elif count > 0:
                x, y, _, _ = self.canvas.coords(self.image)

                # Adjust the position of the circle based on its index
                circle_x = x - 30 + index * 20
                circle_y = y + self.size + 5

                # Create or update the circle position
                if circle:
                    self.canvas.coords(circle, circle_x - 10, circle_y, circle_x + 10, circle_y + 20)
                else:
                    circle = self.canvas.create_oval(circle_x - 10, circle_y, circle_x + 10, circle_y + 20, fill=color)
                    self.pollen_circles[color] = circle

    def get_x(self):
        return self.canvas.coords(self.image)[0]

    def get_y(self):
        return self.canvas.coords(self.image)[1]

    def touches(self, other):
        # Check if the player touches another object
        x1, y1, _, _ = self.canvas.coords(self.image)
        x2, y2, _, _ = self.canvas.coords(other)
        distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return distance < (self.size + 20)  # Assuming flowers have a radius of 20

    def update_pollen_labels(self):
        # Update the pollen inventory labels
        for color, count in self.pollen_inventory.items():
            label = self.pollen_labels.get(color)
            if label:
                self.canvas.delete(label)
            x_position = 50
            y_position = 20 + list(self.pollen_inventory.keys()).index(color) * 20
            self.pollen_labels[color] = self.canvas.create_text(x_position, y_position, text=f'{color.capitalize()}: {count}', anchor=tk.W)

    def collect_pollen(self, color):
        # Increase the count of the corresponding color in the pollen inventory
        self.pollen_inventory[color] += 1
        self.update_pollen_labels()

class Flower:
    def __init__(self, canvas, size=40):
        self.canvas = canvas
        self.size = size
        self.touched = False
        self.colors = ['red', 'blue', 'purple', 'orange', 'pink']
        self.color = random.choice(self.colors)
        x = random.randint(50, 1870)
        y = 1080  # Start the flower at the bottom of the canvas
        self.image = canvas.create_oval(x, y, x + size, y + size, fill=self.color)

    def get_x(self):
        return self.canvas.coords(self.image)[0]

    def get_y(self):
        return self.canvas.coords(self.image)[1]
    
    def move_up(self):
        # Move the flower up the screen
        self.canvas.move(self.image, 0, -1)

    def grow(self):
        # Decrease the size of the flower if it has not been touched
        self.touched = True
        self.size += 10
        self.canvas.coords(self.image, self.get_x(), self.get_y(), self.get_x() + self.size, self.get_y() + self.size)

    def shrink(self):
        # Decrease the size of the flower if it has not been touched
        self.touched = True
        self.size -= 10   
        self.canvas.coords(self.image, self.get_x(), self.get_y(), self.get_x() + self.size, self.get_y() + self.size)

def main():
    root = tk.Tk()
    root.title("Bee Game")

    canvas = tk.Canvas(root, width=1920, height=1080, bg='white')
    canvas.pack()

    player = Player(canvas)
    flowers = []

    def create_flower():
        flower = Flower(canvas)
        flowers.append(flower)
        root.after(1000, create_flower)  # Schedule the next flower creation after 2000 milliseconds (2 seconds)

    create_flower()

    def update_game():
        # Move the player towards the mouse cursor
        player.move_towards(root.winfo_pointerx(), root.winfo_pointery())
        player.move()

        # Check if the player touches any flowers
        for flower in flowers:
            flower.move_up()
            if player.touches(flower.image) and not flower.touched:
                pollen_color = flower.color

                # Check if the player has at least one pollen of the flower's color in their inventory
                if player.pollen_inventory[pollen_color] > 0:
                    # Player has matching pollen, pollenate the flower and make it grow
                    flower.grow()

                    # Decrease the count of the corresponding color in the pollen inventory
                    player.pollen_inventory[pollen_color] -= 1
                    player.update_pollen_labels()
                else:
                    # Player doesn't have matching pollen, take the flower's pollen and shrink the flower
                    flower.shrink()
                    player.collect_pollen(pollen_color)

        root.after(10, update_game)

    update_game()
    root.mainloop()

if __name__ == "__main__":
    main()
