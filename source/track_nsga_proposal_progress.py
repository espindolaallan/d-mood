from EspPipeML import esp_utilities
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np
import matplotlib.pyplot as plt

def plot_front_of_front_3d(h, filename, pareto_lines=False):
    # List to store the Pareto fronts of each generation
    pareto_fronts = []
    all_pareto_points = []

    # Processing the data from each generation
    for generation in h:
        F = generation.get('F')
        I = NonDominatedSorting().do(F, only_non_dominated_front=True)
        pareto_fronts.append(F[I])
        all_pareto_points.append(F[I])  # Adicionar pontos ao array global

    # Join all Pareto fronts into a single array
    all_pareto_points = np.vstack(all_pareto_points)

    # Find the global Pareto front among all data points
    I_global = NonDominatedSorting().do(all_pareto_points, only_non_dominated_front=True)
    global_pareto_points = all_pareto_points[I_global]

    # Setup de cores
    colors = plt.cm.viridis(np.linspace(0, 1, len(pareto_fronts)))

    # Create 3D figure
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the Pareto front for each generation
    for i, pareto_objectives in enumerate(pareto_fronts):
        # ax.scatter(pareto_objectives[:, 0], pareto_objectives[:, 1], pareto_objectives[:, 2], color=colors[i])
        ax.scatter(pareto_objectives[:, 1], pareto_objectives[:, 2], pareto_objectives[:, 0], color=colors[i])

    # Sort global Pareto front
    sorted_indices = np.lexsort((global_pareto_points[:, 2], global_pareto_points[:, 1], global_pareto_points[:, 0]))
    sorted_global_pareto_points = global_pareto_points[sorted_indices]

    # Plotar o front de Pareto global
    if pareto_lines:
        ax.plot(sorted_global_pareto_points[:, 1], sorted_global_pareto_points[:, 2], sorted_global_pareto_points[:, 0], 'r--', linewidth=1, label='Pareto of Paretos')


    ax.set_xlabel('Diversity')
    ax.set_ylabel('Effort')
    ax.set_zlabel('AUC')
    ax.set_title('Evolution of the Pareto Fronts Over Generations')
    # ax.legend()
    ax.grid(True)
    # plt.show()

    # Salvar o plot como PDF
    plt.savefig(filename, format='pdf')
    plt.close()

def plot_front_of_front_2d(h, filename, pareto_lines=False):
    # List to store the Pareto fronts of each generation
    pareto_fronts = []
    all_pareto_points = []

    # Processing the data from each generation
    for generation in h:
        F = generation.get('F')
        I = NonDominatedSorting().do(F, only_non_dominated_front=True)
        pareto_fronts.append(F[I])
        all_pareto_points.append(F[I])  # Add points to the global array

    # Join all points into a single array
    all_pareto_points = np.vstack(all_pareto_points)

    # Find the global Pareto front among all collected points
    I_global = NonDominatedSorting().do(all_pareto_points, only_non_dominated_front=True)
    global_pareto_points = all_pareto_points[I_global]

    # Settings for colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(pareto_fronts)))

    # Create a single figure and axes
    plt.figure(figsize=(20, 14))

    # Plot the Pareto front for each generation
    for i, pareto_objectives in enumerate(pareto_fronts):
        plt.scatter(pareto_objectives[:, 1], pareto_objectives[:, 0], color=colors[i])

    # Sort the points of the global front for coherent plotting
    sorted_indices = np.lexsort((global_pareto_points[:, 0], global_pareto_points[:, 1]))
    sorted_global_pareto_points = global_pareto_points[sorted_indices]

    # Plot the "Pareto of Paretos" with a line
    if pareto_lines:
        plt.plot(sorted_global_pareto_points[:, 1], sorted_global_pareto_points[:, 0], 'r--', linewidth=1, label='Pareto of Paretos')

    # Settings for axes
    plt.xlabel('Diversity')
    plt.ylabel('AUC')
    plt.title('Evolution of the Pareto Front over Generations')
    # plt.legend()
    plt.grid(True)
    # plt.show()

    # Salvar o plot como PDF
    plt.savefig(filename, format='pdf')
    plt.close()

h = esp_utilities.load_from_pickle('../results/nsga2/checkpoint/feature_selection/unsw-nb15_proposal_auc_history.pkl')


path = '../results/nsga2/feature_selection/proposal/'


file_name = 'pareto_fronts_2d_no_lines.pdf'
plot_front_of_front_2d(h, path + file_name, pareto_lines=False)

file_name = 'pareto_fronts_2d_with_lines.pdf'
plot_front_of_front_2d(h, path + file_name, pareto_lines=True)

file_name = 'pareto_fronts_3d_no_lines.pdf'
plot_front_of_front_3d(h, path + file_name, pareto_lines=False)

file_name = 'pareto_fronts_3d_with_lines.pdf'
plot_front_of_front_3d(h, path + file_name, pareto_lines=True)