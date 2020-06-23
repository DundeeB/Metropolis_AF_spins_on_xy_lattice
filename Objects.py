from sklearn.neighbors import radius_neighbors_graph
import numpy as np
import random


class Particle:

    def __init__(self, xy, s, nearest_neighbors, z):
        self.xy, self.s, self.nearest_neighbors, self.z = \
            xy, s, nearest_neighbors, z

    def de(self):
        """
        :return: e_new - e_old where s_new=-s_old.
        """
        energy = 0.0
        for neighbor in self.nearest_neighbors:
            energy -= self.s * neighbor.s
        return -2 * energy

    def flip(self):
        self.s = -self.s


class Configuration:

    def __init__(self, J, sig, positions, boundaries):
        self.J, self.sig, self.positions, self.boundaries = \
            J, sig, positions, boundaries
        self.N = len(positions)

        particles = []
        for p in positions:
            s = 1 if p[2] > self.boundaries[2] / 2 else -1
            particles.append(Particle(xy=p[:2], s=s, nearest_neighbors=[], z=p[2]))
        self.particles = particles

        cyc = lambda p1, p2: Configuration.cyclic_dist(boundaries[:2], p1, p2)
        self.graph = radius_neighbors_graph([p[:2] for p in self.positions], self.sig,
                                            metric=cyc)
        for i in range(len(self.positions)):
            self.particles[i].nearest_neighbors = [self.particles[j] for j in self.graph.getrow(i).indices]

        E = 0
        for p in self.particles:
            for p_ in p.nearest_neighbors:
                E -= self.J * p.s * p_.s / 2  # double counting bonds
        self.E = E

        M = 0
        for p in self.particles: M += p.s
        self.M = M

    @staticmethod
    def cyclic_dist(boundaries, p1, p2):
        dx = np.array(p1) - p2  # direct vector
        dsq = 0
        for i in range(len(p1)):
            L = boundaries[i]
            dsq += min(dx[i] ** 2, (dx[i] + L) ** 2, (dx[i] - L) ** 2)  # find shorter path through B.D.
        return np.sqrt(dsq)

    @property
    def spins(self):
        return [p.s for p in self.particles]

    def Metropolis_flip(self):
        p = self.particles[random.randint(0, len(self.particles) - 1)]
        de = self.J * p.de()  # e_new-e_old
        A = min(1, np.exp(-de))
        u = random.random()
        if u <= A: self.flip(p, de)

    def flip(self, p, de):
        self.E += de
        self.M += -2 * p.s
        p.flip()
