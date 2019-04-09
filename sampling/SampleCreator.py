from skimage import draw
from skimage.draw import circle
import math
import numpy as np
import random


class SampleCreator:
    def __init__(self, img_size=(8, 8), min_obj_size=3, max_obj_size=5):
        if any([max_obj_size > dim for dim in img_size]):
            raise ValueError("Maximum object size should be less than image size")
        self._img_size = img_size
        self._min_obj_size = min_obj_size
        self._max_obj_size = max_obj_size
        self.available_figures = {
            "square": 0,
            "triangle": 1,
            "circle": 2,
            "dot": None
        }
    
    def get_square_img(self):
        img = np.zeros(self._img_size)

        w = np.random.randint(self._min_obj_size, self._max_obj_size)
        h = np.random.randint(self._min_obj_size, self._max_obj_size)
        x = random.randint(0, self._img_size[0] - w)
        y = random.randint(0, self._img_size[1] - h)

        img[x:x+w, y:y+h] += np.ones((w, h))
        return (img, (x, y, w, h, self.available_figures["square"]))
    
    def get_triangle_img(self):
        img = np.zeros(self._img_size)

        s = np.random.randint(self._min_obj_size, self._max_obj_size)
        x = random.randint(0, self._img_size[0] - s)
        y = random.randint(0, self._img_size[1] - s)

        triangle = np.ones((s, s))
        triangle = np.tril(triangle) if random.randint(0, 1) else np.triu(triangle)
        img[x:x+s, y:y+s] += triangle
        return (img, (x, y, s, s, self.available_figures["triangle"]))
    
    def get_dot_img(self):
        img = np.zeros(self._img_size)
        
        x = random.randint(0, self._img_size[0]-1)
        y = random.randint(0, self._img_size[1]-1)
        
        img[x, y] = 1
        return img
    
    def get_circle_img(self):
        img = np.zeros(self._img_size)
        
        s = np.random.randint(self._min_obj_size, self._max_obj_size)
        radius = int(math.floor(s / 2))
        r = random.randint(radius, self._img_size[0] - radius)
        c = random.randint(radius, self._img_size[1] - radius)
        
        rr, cc = circle(r, c, radius, shape=(self._img_size))
        img[rr, cc] = 1
        return (img, (r-radius, c-radius, radius * 2, radius * 2, self.available_figures["circle"]))
    
    def create_sample(self, objs=["square"], fake_objs=["triangle"], max_interception=1, fake_prob=0.5):
        """Create sample image and bounding boxes"""
        if set(objs + fake_objs) - set(self.available_figures.keys()):
            raise ValueError("There is unregistered figure in object list")
        while True:
            sample = np.zeros(self._img_size)
            bboxes = np.zeros((len(objs), 5))
            for i in range(len(objs)):
                if objs[i] == "square":
                    square = self.get_square_img()
                    sample += square[0]
                    bboxes[i] = square[1]
                elif objs[i] == "triangle":
                    triangle = self.get_triangle_img()
                    sample += triangle[0]
                    bboxes[i] = triangle[1]
                elif objs[i] == "circle":
                    circle = self.get_circle_img()
                    sample += circle[0]
                    bboxes[i] = circle[1]
            for i in range(len(fake_objs)):
                if random.random() > fake_prob:
                    break
                if fake_objs[i] == "square":
                    sample += self.get_square_img()[0]
                elif fake_objs[i] == "triangle":
                    sample += self.get_triangle_img()[0]
                elif fake_objs[i] == "circle":
                    sample += self.get_circle_img()[0]
                elif fake_objs[i] == "dot":
                    sample += self.get_dot_img()
            if not len([i for i in sample.flatten() if i > 1]) > max_interception:
                break
        return (sample.astype(bool).astype(int), bboxes)