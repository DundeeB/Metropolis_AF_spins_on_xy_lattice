from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph
import numpy as np
import random
import os
import scipy


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

    def __init__(self, positions, boundaries, J=0, random_initialization=True, sig=None, k=None):
        self.J, self.sig, self.positions, self.boundaries = J, sig, positions, boundaries
        self.N = len(positions)

        particles = []
        for p in positions:
            if random_initialization:
                s = 2 * random.randint(0, 1) - 1
            else:
                s = 1 if p[2] > self.boundaries[2] / 2 else -1
            particles.append(Particle(xy=p[:2], s=s, nearest_neighbors=[], z=p[2]))
        self.particles = particles

        cyc = lambda p1, p2: Configuration.cyclic_dist(boundaries[:2], p1, p2)
        if (k is None) and (sig is None):
            raise ValueError("Must choose k or sig for graph construction")
        if (k is not None) and (sig is not None):
            raise ValueError("Must choose only one of  k and sig for graph construction")
        if k is not None:
            self.graph = kneighbors_graph([p[:2] for p in self.positions], n_neighbors=k, metric=cyc)
            I, J, _ = scipy.sparse.find(self.graph)[:]
            Ed = [(i, j) for (i, j) in zip(I, J)]
            Eud = []
            udgraph = scipy.sparse.csr_matrix((self.N, self.N))
            for i, j in Ed:
                if ((j, i) in Ed) and ((i, j) not in Eud) and ((j, i) not in Eud):
                    Eud.append((i, j))
                    udgraph[i, j] = 1
                    udgraph[j, i] = 1
            self.graph = udgraph
        else:
            self.graph = radius_neighbors_graph([p[:2] for p in self.positions], self.sig, metric=cyc)
        for i in range(len(self.positions)):
            self.particles[i].nearest_neighbors = [self.particles[j] for j in self.graph.getrow(i).indices]

        self.E = 0
        for p in self.particles:
            for p_ in p.nearest_neighbors:
                self.E -= self.J * p.s * p_.s / 2  # double counting bonds

        self.M = 0
        for p in self.particles:
            H = boundaries[2]
            self.M += p.s if p.z > H / 2 else -p.s

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
        if u <= A:
            self.E += de
            H = self.boundaries[2]
            self.M += 2 * p.s * (-1 if p.z > H / 2 else 1)
            p.flip()

    def anneal(self, iterations, dTditer=0, save_diter=1):
        M, E, J = [], [], []
        T = 1/self.J
        for i in range(iterations):
            if i % save_diter == 0:
                M.append(self.M)
                E.append(self.E)
                J.append(self.J)
            self.Metropolis_flip()
            T += dTditer
            self.J = 1/T
        return E, J, M


class FilesHandler:

    def __init__(self, orig_dir, sp_name, output_dir):
        self.output_dir = output_dir
        self.positions = np.loadtxt(os.path.join(orig_dir, sp_name))
        np.savetxt(os.path.join(output_dir, sp_name), self.positions)

    def dump_spins(self, configuration, name):
        np.savetxt(os.path.join(self.output_dir, name), configuration.spins)

    def append_E_M(self, E, M, i0=0):
        I = [i + i0 for i in range(len(E))]
        with open(os.path.join(self.output_dir, 'M_E_output'), 'ab') as f:
            np.savetxt(f, np.transpose([I, E, M]))
