# https://picos-api.gitlab.io/picos/graphs.html#maxcut-relaxation-sdp
import picos as pic
import networkx as nx
import pylab
import cvxopt as cvx
import cvxopt.lapack
import numpy as np


def find_max_cut(graph, positions):
    # Move G to LCF notation.
    G = nx.convert_matrix.from_scipy_sparse_matrix(graph)
    pos = {i: positions[i] for i in range(len(positions))}

    # Generate edge capacities.
    c = {}
    for e in sorted(G.edges(data=True)):
        capacity = 1
        e[2]['capacity'] = capacity
        c[(e[0], e[1])] = capacity
        c[(e[1], e[0])] = capacity

    # Convert the capacities to a PICOS expression.
    cc = pic.new_param('c', c)

    # Set source and sink nodes for flow computation.
    s = 16
    t = 10

    # Set node colors.
    N = len(positions)
    node_colors = ['lightgrey'] * N
    node_colors[s] = 'lightgreen'  # Source is green.
    node_colors[t] = 'lightblue'  # Sink is blue.

    # Define a plotting helper that closes the old and opens a new figure.
    def new_figure():
        try:
            global fig
            pylab.close(fig)
        except NameError:
            pass
        fig = pylab.figure(figsize=(11, 8))
        fig.gca().axes.get_xaxis().set_ticks([])
        fig.gca().axes.get_yaxis().set_ticks([])

    # Plot the graph with the edge capacities.
    new_figure()
    nx.draw_networkx(G, pos, node_color=node_colors)
    labels = {
        e: '{} | {}'.format(c[(e[0], e[1])], c[(e[1], e[0])])
        for e in G.edges if e[0] < e[1]}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    pylab.show()

    # Make G undirected.
    G = nx.Graph(G)

    # Allocate weights to the edges.
    for (i, j) in G.edges():
        G[i][j]['weight'] = 1

    maxcut = pic.Problem()

    # Add the symmetric matrix variable.
    X = maxcut.add_variable('X', (N, N), 'symmetric')

    # Retrieve the Laplacian of the graph.
    LL = 1 / 4. * nx.laplacian_matrix(G).todense()
    L = pic.new_param('L', LL)

    # Constrain X to have ones on the diagonal.
    maxcut.add_constraint(pic.diag_vect(X) == 1)

    # Constrain X to be positive semidefinite.
    maxcut.add_constraint(X >> 0)

    # Set the objective.
    maxcut.set_objective('max', L | X)

    # print(maxcut)

    # Solve the problem.
    maxcut.solve(solver='cvxopt')

    # print('bound from the SDP relaxation: {0}'.format(maxcut.obj_value()))

    # Use a fixed RNG seed so the result is reproducable.
    cvx.setseed(1)

    # Perform a Cholesky factorization.
    V = X.value
    cvxopt.lapack.potrf(V)
    for i in range(N):
        for j in range(i + 1, N):
            V[i, j] = 0

    # Do up to 100 projections. Stop if we are within a factor 0.878 of the SDP
    # optimal value.
    count = 0
    obj_sdp = maxcut.obj_value()
    obj = 0
    while (count < 100 or obj < 0.878 * obj_sdp):
        r = cvx.normal(20, 1)
        x = cvx.matrix(np.sign(V * r))
        o = (x.T * L * x).value
        if o > obj:
            x_cut = x
            obj = o
        count += 1
    x = x_cut

    # Extract the cut and the seperated node sets.
    S1 = [n for n in range(N) if x[n] < 0]
    S2 = [n for n in range(N) if x[n] > 0]
    cut = [(i, j) for (i, j) in G.edges() if x[i] * x[j] < 0]
    leave = [e for e in G.edges if e not in cut]

    # Close the old figure and open a new one.
    new_figure()

    # Assign colors based on set membership.
    node_colors = [('lightgreen' if n in S1 else 'lightblue') for n in range(N)]

    # Draw the nodes and the edges that are not in the cut.
    nx.draw_networkx(G, pos, node_color=node_colors, edgelist=leave)
    labels = {e: '{}'.format(G[e[0]][e[1]]['weight']) for e in leave}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    # Draw the edges that are in the cut.
    nx.draw_networkx_edges(G, pos, edgelist=cut, edge_color='r')
    labels = {e: '{}'.format(G[e[0]][e[1]]['weight']) for e in cut}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='r')

    # Show the relaxation optimum value and the cut capacity.
    rval = maxcut.obj_value()
    sval = sum(G[e[0]][e[1]]['weight'] for e in cut)
    fig.suptitle(
        'SDP relaxation value: {0:.1f}\nCut value: {1:.1f} = {2:.3f}×{0:.1f}'
            .format(rval, sval, sval / rval), fontsize=16, y=0.97)

    # Show the figure.
    pylab.show()

    return S1, S2
