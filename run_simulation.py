from Objects import *
import matplotlib.pyplot as plt
import matplotlib

orid_dir = "../OOP_hard_sphere_event_chain"
sp_name = "53647110"
files_handler = FilesHandler(orid_dir, sp_name, 'debug')

J = 0
sig = 2
iterations = int(1e4)
save_once_in = int(1e3)

state = Configuration(J, sig, files_handler.positions,
                      [max([p[i] for p in files_handler.positions]) + sig / 2 for i in [0, 1, 2]])
files_handler.dump_spins(state, '0')

M = np.zeros(iterations)
E = np.zeros(iterations)
for i in range(iterations):
    M[i] = state.M
    E[i] = state.E
    state.Metropolis_flip()
    if i % save_once_in == 0:
        files_handler.dump_spins(state, str(i))
files_handler.append_E_M(E, M, 0)

plt.subplot(2, 1, 1)
plt.plot(E)
plt.ylabel("Energy")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(M)
plt.xlabel("iteration number")
plt.ylabel("Magnetization sum(s in A)-sum(s in B)")
plt.grid()
plt.ylim([0, max(M)])

font = {'family': 'normal',
        'weight': 'bold',
        'size': 16}

matplotlib.rc('font', **font)
plt.show(block=True)
