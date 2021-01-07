from Objects import *
import matplotlib.pyplot as plt
import matplotlib

orid_dir = "../post_process/from_ATLAS3.0/N=10000_h=0.8_rhoH=0.8_AF_square_ECMC/"
sp_name = "22906182"
files_handler = FilesHandler(orid_dir, sp_name, 'debug')

J = -10  # / 2.269 + 0.15
iterations = int(1e5)
save_diter = iterations / int(1e4)
for random_initialization in [True, False]:
    state = Configuration(files_handler.positions,
                          [max([p[i] for p in files_handler.positions]) + 1 for i in [0, 1, 2]], k=4, J=-0.1,
                          random_initialization=random_initialization)

    _, _, M = state.anneal(iterations, save_diter=save_diter, dTditer=1 / J / iterations)

    # font
    size = 15
    params = {'legend.fontsize': 'large',
              'figure.figsize': (20, 8),
              'axes.labelsize': size,
              'axes.titlesize': size,
              'xtick.labelsize': size * 0.75,
              'ytick.labelsize': size * 0.75,
              'axes.titlepad': 25}
    plt.rcParams.update(params)
    x = np.array(range(len(M))) * save_diter
    plt.plot(x, np.abs(M))
    plt.xlabel("iteration number")
    plt.ylabel("Magnetization |sum(s in A)-sum(s in B)|")
    plt.grid()
plt.show()
