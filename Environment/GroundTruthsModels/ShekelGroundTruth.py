
import numpy as np
from deap import benchmarks

import sys

sys.path.append('.')

class GroundTruth(object):

    """ Ground Truth generator class.
        It creates a ground truth within the specified navigation map.
        The ground truth is generated randomly following some realistic rules of the enqqqviornment
        and using a Shekel function.
    """

    def __init__(self, grid, max_number_of_peaks=None, is_bounded = True, seed = 0, peaks_location = 'Random'):

        """ Maximum number of peaks encountered in the scenario. """
        self.max_number_of_peaks = 6 if max_number_of_peaks is None else max_number_of_peaks
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed) # random number generator, it's better than set a np.random.seed() (https://builtin.com/data-science/numpy-random-seed)
        self.rng_seed_for_steps = np.random.default_rng(seed=self.seed+1)
        self.rng_steps = np.random.default_rng(seed=self.rng_seed_for_steps.integers(0, 1000000))
        self.peaks_location = peaks_location

        """ random map features creation """
        self.grid = 1.0 - grid
        self.xy_size = np.asarray(grid.shape)
        self.xy_size = np.flip(self.xy_size)
        self.is_bounded = is_bounded
        self.dt = 0.01

        # Peaks positions bounded from 1 to 9 in every axis
        self.peaks_low_limit_dict = {'Upper': [0.1,0], 'MiddleLeft': [0.10,0.45], 'MiddleRight': [0.55,0.4], 'Middle': [0.35,0.45], 'Bottom': [0.67,0.7]}
        self.peaks_high_limit_dict = {'Upper': [0.6, 0.3], 'MiddleLeft': [0.3, 0.65], 'MiddleRight': [0.8, 0.6], 'Middle': [0.65, 0.62], 'Bottom': [0.9, 0.9]}
        self.number_of_peaks = self.rng.integers(1, self.max_number_of_peaks+1)
        if self.peaks_location == 'Random':
            self.A = self.rng.random((self.number_of_peaks, 2)) * self.xy_size
        else: # Delimit peaks
            self.A = np.array([[self.rng.uniform(low=self.peaks_low_limit_dict[self.peaks_location][0], high=self.peaks_high_limit_dict[self.peaks_location][0], size=None), self.rng.uniform(low=self.peaks_low_limit_dict[self.peaks_location][1], high=self.peaks_high_limit_dict[self.peaks_location][1], size=None)] for _ in range(self.number_of_peaks)]) * self.xy_size
        # Peaks size bounded from a minimum 2.5 to 5
        self.C = self.rng.random((self.number_of_peaks, 1)) + 0.5

        """ Creation of the map field """
        self._x = np.arange(0, self.grid.shape[1], 1)
        self._y = np.arange(0, self.grid.shape[0], 1)

        self._x, self._y = np.meshgrid(self._x, self._y)

        self._z, self.meanz, self.stdz, self.normalized_z = None, None, None, None # To instantiate attr after assigning in __init__
        self.create_field()  # This method creates the normalized_z values

    def shekel_arg0(self, sol):

        return np.nan if self.grid[sol[1]][sol[0]] == 1 else \
            benchmarks.shekel(sol[:2], self.A, self.C)[0]

    def create_field(self):

        """ Creation of the normalized z field """
        self._z = np.fromiter(map(self.shekel_arg0, zip(self._x.flat, self._y.flat)), dtype=np.float32,
                              count=self._x.shape[0] * self._x.shape[1]).reshape(self._x.shape)

        self.meanz = np.nanmean(self._z)
        self.stdz = np.nanstd(self._z)

        if self.stdz > 0.001:
            self.normalized_z = (self._z - self.meanz) / self.stdz
        else:
            self.normalized_z = self._z

        if self.is_bounded:
            self.normalized_z = np.nan_to_num(self.normalized_z, nan=np.nanmin(self.normalized_z))
            self.normalized_z = (self.normalized_z - np.min(self.normalized_z))/(np.max(self.normalized_z) - np.min(self.normalized_z))

        #self.normalized_z = self.normalized_z.T

    def reset(self):
        """ Reset ground Truth """
        # Peaks positions bounded from 1 to 9 in every axis
        self.number_of_peaks = self.rng.integers(1,self.max_number_of_peaks+1)
        if self.peaks_location == 'Random':
            self.A = self.rng.random((self.number_of_peaks, 2)) * self.xy_size
        else: # Delimit peaks
            self.A = np.array([[self.rng.uniform(low=self.peaks_low_limit_dict[self.peaks_location][0], high=self.peaks_high_limit_dict[self.peaks_location][0], size=None), self.rng.uniform(low=self.peaks_low_limit_dict[self.peaks_location][1], high=self.peaks_high_limit_dict[self.peaks_location][1], size=None)] for _ in range(self.number_of_peaks)]) * self.xy_size
        # Peaks size bounded from a minimum 2.5 to 5
        self.C = 10*(self.rng.random((self.number_of_peaks, 1)) + 0.5)
        # Reconstruct the field #
        self.create_field()
        # New seed for steps #
        self.rng_steps = np.random.default_rng(seed=self.rng_seed_for_steps.integers(0, 1000000))

    def read(self, position=None):

        """ Read the complete ground truth or a certain position """

        if position is None:
            return self.normalized_z
        else:
            return self.normalized_z[position[0]][position[1]]

    def render(self):

        """ Show the ground truth """
        plt.imshow(self.read(), cmap='inferno', interpolation='none')
        cs = plt.contour(self.read(), colors='royalblue', alpha=1, linewidths=1)
        plt.clabel(cs, inline=1, fontsize=7)
        plt.title("Nº of peaks: {}".format(gt.number_of_peaks), color='black', fontsize=10)
        im = plt.plot(self.A[:, 0],
                      self.A[:, 1], 'hk', )
        plt.show()

    def step(self):
        """ Move every maximum with a random walk noise """

        self.A += self.dt*(2*(self.rng_steps.random([*self.A.shape])-0.5) * self.xy_size * 0.9 + self.xy_size*0.1)
        # self.A += self.dt*(2*(np.random.rand(*self.A.shape)-0.5) * self.xy_size * 0.9 + self.xy_size*0.1)
        self.create_field()

        pass


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    ypacarai_map = np.genfromtxt('Environment/Maps/ypacarai_map_low_res.csv',delimiter=',',dtype=float)
    gt = GroundTruth(ypacarai_map, max_number_of_peaks=6, is_bounded=True, seed=10, peaks_location='MiddleRight')

    # for i in range(100):
    #     gt.reset()
    #     gt.render()

    A_total = np.empty((0, 2), float)

    for i in range(100):
        gt.reset()
        A_total = np.vstack((A_total,gt.A))
    
    plt.imshow(ypacarai_map)
    im = plt.plot(A_total[:, 0], A_total[:, 1], 'rx', )
    plt.show()





