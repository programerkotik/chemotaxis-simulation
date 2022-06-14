import numpy as np


import numpy as np

class Cell():
    def __init__(self, x, y, self_generated, k_max, V_max):
        self.x = x
        self.y = y
        self.self_generated = self_generated
        self.k_max = k_max
        self.V_max = V_max


    def sense(self, phi, gridmap):
        height, width = phi.shape
        concentrations = []
        coordinates = []
    
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == dx == 0: continue
                y = self.y + dy
                x = self.x + dx

                # map constraints
                if 0 <= y < height and 0 <= x < width and gridmap[y,x] == 0:

                    concentrations.append(phi[y, x])
                    coordinates.append([y, x])
                    
        return np.array(coordinates), np.array(concentrations)
        
    def step(self, phi, gridmap):
        coordinates, concentrations = self.sense(phi, gridmap)
        concentrations_sum = concentrations.sum()
        
        if concentrations_sum == 0: 
            return
        
        probabilities = concentrations / concentrations_sum
        i = np.random.choice(len(coordinates), p=probabilities)
        self.y, self.x = coordinates[i]

        if self.self_generated:
            self.consume(phi, self.k_max, self.V_max)

    # for self generated chemotaxis
    def consume(self, phi, k_max, V_max):
        c = phi[self.y, self.x]
        rate = V_max * c / (c + k_max)
        phi[self.y, self.x] -= rate * c