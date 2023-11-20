import numpy as np
from scipy.ndimage import gaussian_filter, convolve
import matplotlib.colors
import matplotlib.pyplot as plt

algae_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["dodgerblue","darkcyan", "darkgreen", "forestgreen"])
background_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["sienna","sienna"])
fuelspill_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["dodgerblue", "olive", "saddlebrown", "indigo"])


class algae_bloom:

    def __init__(self, grid: np.ndarray, dt = 0.2, seed = 0) -> None:
        """ Generador de ground truths de algas con dinámica """

        np.random.seed(seed)
        # Creamos un mapa vacio #
        self.map = np.zeros_like(grid)
        self.grid = grid
        self.particles = None
        self.starting_point = None
        self.visitable_positions = np.column_stack(np.where(grid == 1))
        self.fig = None
        self.dt = dt
        self.current_field_fn = np.vectorize(self.current_field, signature="(n) -> (n)")
        self.apply_bounds_fn = np.vectorize(self.apply_bounds, signature="(n) -> (n)")

        #self.contour_currents_x = convolve(self.grid, np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,1,-1,-2],[0,0,0,0,0],[0,0,0,0,0]]), mode='constant')
        #self.contour_currents_y = convolve(self.grid, np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,-1,0,0],[0,0,-2,0,0]]), mode='constant')
        self.contour_currents_x = convolve(self.grid, np.array([[0,0,0],[0,1,-1],[0,0,0]]), mode='constant')*2
        self.contour_currents_y = convolve(self.grid, np.array([[0,0,0],[0,1,0],[0,-1,0]]), mode='constant')*2
    def reset(self):

        starting_point = np.array((np.random.randint(self.map.shape[0]/4, 3*self.map.shape[0]/4), np.random.randint(self.map.shape[1]/2, 2* self.map.shape[1]/3)))
        self.particles = np.random.multivariate_normal(starting_point, np.array([[7.0, 0.0],[0.0, 7.0]]),size=(100,))
        
        starting_point = np.array((np.random.randint(self.map.shape[0]/4, 3*self.map.shape[0]/4), np.random.randint(self.map.shape[1]/2, 2* self.map.shape[1]/3)))
        self.particles = np.vstack(( self.particles, np.random.multivariate_normal(starting_point, np.array([[3.0, 0.0],[0.0, 3.0]]),size=(100,))))

        self.in_bound_particles = np.array([particle for particle in self.particles if self.is_inside(particle)])
        self.map[self.in_bound_particles[:,0].astype(int), self.in_bound_particles[:, 1].astype(int)] = 1.0

        self.algae_map = gaussian_filter(self.map, 0.8)
        
        return self.algae_map

    def apply_bounds(self, position):
        
        new_position = np.clip(position, (0,0), np.array(self.map.shape)-1)

        new_position[0] -= self.contour_currents_x[int(position[0]), int(position[1])]
        new_position[1] -= self.contour_currents_y[int(position[0]), int(position[1])]

        return new_position

        
    def current_field(self, position):

        #u = - np.sin(2 * np.pi * (position[0] - self.map.shape[0] // 2) / self.map.shape[0]) + np.cos(2 * np.pi * (position[1] - self.map.shape[1] // 2) / self.map.shape[1])
        #v = np.cos(2 * np.pi * (position[0] - self.map.shape[0] // 2) / self.map.shape[0]) - np.sin(2 * np.pi * (position[1] - self.map.shape[1] // 2) / self.map.shape[1])
        
        if self.contour_currents_x[int(position[0]), int(position[1])] == 0.0 or self.contour_currents_y[int(position[0]), int(position[1])] == 0.0:

            u = -(position[1] - self.map.shape[1] / 2) / np.linalg.norm(position - np.array(self.map.shape)/2 + 1e-6) + np.random.rand()
            v = (position[0] - self.map.shape[0] / 2) / np.linalg.norm(position - np.array(self.map.shape)/2 + 1e-6) + np.random.rand()

        else:
            u,v = 0,0

        u,v = np.clip((u,v), -1.0, 1.0)
        
        
        return np.array((u*np.random.rand(),v*np.random.rand()))

    def is_inside(self, particle):

        
        particle = particle.astype(int)
        if particle[0] >= 0 and particle[0] < self.map.shape[0] and  particle[1] >= 0 and particle[1] < self.map.shape[1] and self.grid[particle[0], particle[1]] == 1:

            return True
        else:
            return False

    def step(self):

        self.map[:,:] = 0.0
        
        current_movement = self.current_field_fn(self.in_bound_particles)

        self.in_bound_particles = self.apply_bounds_fn(self.in_bound_particles)
        self.in_bound_particles = self.in_bound_particles + self.dt * current_movement
        
        self.in_bound_particles = np.array([particle for particle in self.in_bound_particles if self.is_inside(particle)])

        self.map[self.in_bound_particles[:,0].astype(int), self.in_bound_particles[:, 1].astype(int)] = 1.0

        self.algae_map = gaussian_filter(self.map, 0.8) * self.grid

        return self.algae_map

    def render(self):
        
        f_map = self.algae_map
        f_map[self.grid == 0] = np.nan

        if self.fig is None:
            current = self.current_field_fn(self.visitable_positions)
            self.fig, self.ax = plt.subplots(1,1)
            self.ax.quiver(self.visitable_positions[::6,1], self.visitable_positions[::6,0], current[::6,1], -current[::6,0], color='black', alpha = 0.25)
            self.d = self.ax.imshow(f_map, cmap = algae_colormap, vmin=0.0, vmax = 1.0)
            
            background = self.grid.copy()
            background[background == 1] = np.nan
            self.ax.imshow(background, cmap=background_colormap)
            
        else:
            self.d.set_data(f_map)

        self.fig.canvas.draw()
        plt.pause(0.01)
    
    def read(self):

        return self.algae_map

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    gt = algae_bloom(np.genfromtxt('../Maps/example_map.csv', delimiter=','), dt=0.2)

    m = gt.reset()
    gt.render()

    for _ in range(1000):

        m = gt.step()
        gt.render()

    


        
        
        