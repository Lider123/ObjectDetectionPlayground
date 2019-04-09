from .SampleCreator import SampleCreator
import numpy as np
import random


class RGBSampleCreator(SampleCreator):
    def random_rgbcolor(self):
        return list(np.random.choice(range(256), size=3))
    
    def get_square_img(self):
        result = super(RGBSampleCreator, self).get_square_img()
        square = result[0]
        color = self.random_rgbcolor()
        square = [color if px != 0 else [0, 0, 0] for row in square for px in row]
        square = np.array(square).reshape(self._img_size + (3,))
        return (square,) + result[1:]
    
    def get_triangle_img(self):
        result = super(RGBSampleCreator, self).get_triangle_img()
        triangle = result[0]
        color = self.random_rgbcolor()
        triangle = [color if px != 0 else [0, 0, 0] for row in triangle for px in row]
        triangle = np.array(triangle).reshape(self._img_size + (3,))
        return (triangle,) + result[1:]
    
    def get_circle_img(self):
        result = super(RGBSampleCreator, self).get_circle_img()
        circle = result[0]
        color = self.random_rgbcolor()
        circle = [color if px != 0 else [0, 0, 0] for row in circle for px in row]
        circle = np.array(circle).reshape(self._img_size + (3,))
        return (circle,) + result[1:]
    
    def get_dot_img(self):
        img = np.zeros(self._img_size + (3,))
        
        x = random.randint(0, self._img_size[0]-1)
        y = random.randint(0, self._img_size[1]-1)
        color = self.random_rgbcolor()
        
        img[x, y] = color
        return img
    
    def interception(self, img1, img2):
        if img1.shape != img2.shape:
            raise ValueError("Images must have equal shape")
        shape = self._img_size + (3,)
        img1 = img1.reshape(shape[0] * shape[1], shape[2])
        img2 = img2.reshape(shape[0] * shape[1], shape[2])
        img1 = [bool(sum(px)) for px in img1]
        img2 = [bool(sum(px)) for px in img2]
        return sum([px1 & px2 for px1, px2 in zip(img1, img2)])
    
    def concatim(self, img1, img2):
        if img1.shape != img2.shape:
            raise ValueError("Images must have equal shape")
        shape = self._img_size + (3,)
        img1 = img1.reshape(shape[0] * shape[1], shape[2])
        img2 = img2.reshape(shape[0] * shape[1], shape[2])
        img_res = [px1 if sum(px2) == 0 else px2 for px1, px2 in zip(img1, img2)]
        return np.array(img_res).reshape(shape)
        
    def create_sample(self, objs=["square"], fake_objs=["triangle"], max_interception=1, fake_prob=0.5):
        """Create sample image and bounding boxes"""
        if set(objs + fake_objs) - set(self.available_figures.keys()):
            raise ValueError("There is unregistered figure in object list")
        sample = np.zeros(self._img_size + (3,))
        bboxes = np.zeros((len(objs), 4))
        classes = np.zeros((len(objs)))
        random.shuffle(objs)
        for i in range(len(objs)):
            while True:
                if objs[i] == "square":
                    img = self.get_square_img()
                elif objs[i] == "triangle":
                    img = self.get_triangle_img()
                elif objs[i] == "circle":
                    img = self.get_circle_img()
                if not self.interception(sample, img[0]) > max_interception:
                    sample = self.concatim(sample, img[0])
                    bboxes[i] += img[1][:-1]
                    classes[i] += img[1][-1]
                    break
        for i in range(len(fake_objs)):
            while True:
                if fake_objs[i] == "square":
                    img = self.get_square_img()
                elif fake_objs[i] == "triangle":
                    img = self.get_triangle_img()
                elif fake_objs[i] == "circle":
                    img = self.get_circle_img()
                elif fake_objs[i] == "dot":
                    img = self.get_dot_img()
                if not self.interception(sample, img[0]) > max_interception:
                    sample = self.concatim(sample, img[0])
                    break
        return (sample, bboxes, classes)
