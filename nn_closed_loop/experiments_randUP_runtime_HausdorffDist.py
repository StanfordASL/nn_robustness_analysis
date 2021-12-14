import nn_closed_loop.example as ex
import numpy as np
from tabulate import tabulate
import pandas as pd
import datetime
import os
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

import nn_closed_loop.dynamics as dynamics
import nn_closed_loop.analyzers as analyzers
import nn_closed_loop.constraints as constraints
import nn_closed_loop.sampling_based.randUP as randUP
import nn_closed_loop.sampling_based.kernelUP as kernelUP
import nn_closed_loop.sampling_based.GoTube as GoTube
from nn_closed_loop.utils.nn import load_controller


results_dir = "{}/results/logs/".format(
    os.path.dirname(os.path.abspath(__file__))
)
os.makedirs(results_dir, exist_ok=True)


expts = [
    # {
    #     'partitioner': 'None',
    #     'propagator': 'SeparableCROWN',
    #     'sampling_based': False,
    # },
    # {
    #     'partitioner': 'None',
    #     'propagator': 'SeparableSGIBP',
    #     'sampling_based': False,
    # },
    #
    #
    #
    #
    #
    {
        'partitioner': 'None',
        'propagator': 'CROWN',
        'sampling_based': False,
        'boundaries': 'lp'
    },
    {
        'partitioner': 'Uniform',
        'num_partitions': "[4, 4]",
        'propagator': 'CROWN',
        'sampling_based': False,
        'boundaries': 'lp'
    },
    {
        'partitioner': 'None',
        'propagator': 'CROWN',
        'sampling_based': False,
        'boundaries': 'polytope'
    },
    {
        'partitioner': 'Uniform',
        'num_partitions': "[4, 4]",
        'propagator': 'CROWN',
        'sampling_based': False,
        'boundaries': 'polytope'
    },
    # {
    #     'partitioner': 'None',
    #     'propagator': 'SDP',
    #     'cvxpy_solver': 'SCS',
    #     'sampling_based': False,
    # },
    # {
    #     'partitioner': 'Uniform',
    #     'num_partitions': "[4, 4]",
    #     'propagator': 'SDP',
    #     'cvxpy_solver': 'SCS',
    #     'sampling_based': False,
    # },
    # {
    #     'partitioner': 'randUP',
    #     'propagator': 'randUP',
    #     'randUP': True,
    #     'nb_samples': 1000,
    #     'epsilon': 0.02,
    # },
    {
        'partitioner': 'randUP_M100',
        'propagator': 'randUP_M100',
        'sampling_based': True,
        'boundaries': 'hull',
        'randUP': True,
        'kernelUP': False,
        'GoTube': False,
        'nb_samples': 100,
        'epsilon': 0.0,
    },
    {
        'partitioner': 'randUP_M200',
        'propagator': 'randUP_M200',
        'sampling_based': True,
        'boundaries': 'hull',
        'randUP': True,
        'kernelUP': False,
        'GoTube': False,
        'nb_samples': 200,
        'epsilon': 0.0,
    },
    {
        'partitioner': 'randUP_M300',
        'propagator': 'randUP_M300',
        'sampling_based': True,
        'boundaries': 'hull',
        'randUP': True,
        'kernelUP': False,
        'GoTube': False,
        'nb_samples': 300,
        'epsilon': 0.0,
    },
    {
        'partitioner': 'randUP_M500',
        'propagator': 'randUP_M500',
        'sampling_based': True,
        'boundaries': 'hull',
        'randUP': True,
        'kernelUP': False,
        'GoTube': False,
        'nb_samples': 500,
        'epsilon': 0.0,
    },
    {
        'partitioner': 'randUP_M1k',
        'propagator': 'randUP_M1k',
        'sampling_based': True,
        'boundaries': 'hull',
        'randUP': True,
        'kernelUP': False,
        'GoTube': False,
        'nb_samples': 1000,
        'epsilon': 0.0,
    },
    {
        'partitioner': 'randUP_M2k',
        'propagator': 'randUP_M2k',
        'sampling_based': True,
        'boundaries': 'hull',
        'randUP': True,
        'kernelUP': False,
        'GoTube': False,
        'nb_samples': 2000,
        'epsilon': 0.0,
    },
    {
        'partitioner': 'randUP_M3k',
        'propagator': 'randUP_M3k',
        'sampling_based': True,
        'boundaries': 'hull',
        'randUP': True,
        'kernelUP': False,
        'GoTube': False,
        'nb_samples': 3000,
        'epsilon': 0.0,
    },
    {
        'partitioner': 'randUP_M5k',
        'propagator': 'randUP_M5k',
        'sampling_based': True,
        'boundaries': 'hull',
        'randUP': True,
        'kernelUP': False,
        'GoTube': False,
        'nb_samples': 5000,
        'epsilon': 0.0,
    },
    {
        'partitioner': 'randUP_M10k',
        'propagator': 'randUP_M10k',
        'sampling_based': True,
        'boundaries': 'hull',
        'randUP': True,
        'kernelUP': False,
        'GoTube': False,
        'nb_samples': 10000,
        'epsilon': 0.0,
    },
    #
    #
    #
    #
    #
    {
        'partitioner': 'kernelUP_M100',
        'propagator': 'kernelUP_M100',
        'sampling_based': True,
        'boundaries': 'kernel',
        'randUP': False,
        'kernelUP': True,
        'GoTube': False,
        'nb_samples': 100,
        'Lambda': 0.1,
        'sigma': 0.05,
    },
    {
        'partitioner': 'kernelUP_M200',
        'propagator': 'kernelUP_M200',
        'sampling_based': True,
        'boundaries': 'kernel',
        'randUP': False,
        'kernelUP': True,
        'GoTube': False,
        'nb_samples': 200,
        'Lambda': 0.1,
        'sigma': 0.05,
    },
    {
        'partitioner': 'kernelUP_M300',
        'propagator': 'kernelUP_M300',
        'sampling_based': True,
        'boundaries': 'kernel',
        'randUP': False,
        'kernelUP': True,
        'GoTube': False,
        'nb_samples': 300,
        'Lambda': 0.1,
        'sigma': 0.05,
    },
    {
        'partitioner': 'kernelUP_M500',
        'propagator': 'kernelUP_M500',
        'sampling_based': True,
        'boundaries': 'kernel',
        'randUP': False,
        'kernelUP': True,
        'GoTube': False,
        'nb_samples': 500,
        'Lambda': 0.1,
        'sigma': 0.02,
    },
    {
        'partitioner': 'kernelUP_M1k',
        'propagator': 'kernelUP_M1k',
        'sampling_based': True,
        'boundaries': 'kernel',
        'randUP': False,
        'kernelUP': True,
        'GoTube': False,
        'nb_samples': 1000,
        'Lambda': 0.1,
        'sigma': 0.05,
    },
    #
    #
    #
    #
    #
    {
        'partitioner': 'GoTube_M100',
        'propagator': 'GoTube_M100',
        'sampling_based': True,
        'boundaries': 'ball',
        'randUP': False,
        'kernelUP': False,
        'GoTube': True,
        'nb_samples': 100,
        'epsilon': 0.0,
    },
    {
        'partitioner': 'GoTube_M200',
        'propagator': 'GoTube_M200',
        'sampling_based': True,
        'boundaries': 'ball',
        'randUP': False,
        'kernelUP': False,
        'GoTube': True,
        'nb_samples': 200,
        'epsilon': 0.0,
    },
    {
        'partitioner': 'GoTube_M300',
        'propagator': 'GoTube_M300',
        'sampling_based': True,
        'boundaries': 'ball',
        'randUP': False,
        'kernelUP': False,
        'GoTube': True,
        'nb_samples': 300,
        'epsilon': 0.0,
    },
    {
        'partitioner': 'GoTube_M500',
        'propagator': 'GoTube_M500',
        'sampling_based': True,
        'boundaries': 'ball',
        'randUP': False,
        'kernelUP': False,
        'GoTube': True,
        'nb_samples': 500,
        'epsilon': 0.0,
    },
    {
        'partitioner': 'GoTube_M1k',
        'propagator': 'GoTube_M1k',
        'sampling_based': True,
        'boundaries': 'ball',
        'randUP': False,
        'kernelUP': False,
        'GoTube': True,
        'nb_samples': 1000,
        'epsilon': 0.0,
    },
    {
        'partitioner': 'GoTube_M2k',
        'propagator': 'GoTube_M2k',
        'sampling_based': True,
        'boundaries': 'ball',
        'randUP': False,
        'kernelUP': False,
        'GoTube': True,
        'nb_samples': 2000,
        'epsilon': 0.0,
    },
    {
        'partitioner': 'GoTube_M3k',
        'propagator': 'GoTube_M3k',
        'sampling_based': True,
        'boundaries': 'ball',
        'randUP': False,
        'kernelUP': False,
        'GoTube': True,
        'nb_samples': 3000,
        'epsilon': 0.0,
    },
    {
        'partitioner': 'GoTube_M5k',
        'propagator': 'GoTube_M5k',
        'sampling_based': True,
        'boundaries': 'ball',
        'randUP': False,
        'kernelUP': False,
        'GoTube': True,
        'nb_samples': 5000,
        'epsilon': 0.0,
    },
    {
        'partitioner': 'GoTube_M10k',
        'propagator': 'GoTube_M10k',
        'sampling_based': True,
        'boundaries': 'ball',
        'randUP': False,
        'kernelUP': False,
        'GoTube': True,
        'nb_samples': 10000,
        'epsilon': 0.0,
    },
]

class Experiment:
    def __init__(self):
        self.info = {
            ('CROWN', 'Uniform'): {
                'name': 'Reach-LP-Partition',
                'color': 'tab:green',
                'ls': '-',
            },
            ('CROWN', 'None'): {
                'name': 'Reach-LP',
                'color': 'tab:green',
                'ls': '--',
            },
            ('SDP', 'Uniform'): {
                'name': 'Reach-SDP-Partition',
                'color': 'tab:red',
                'ls': '-',
            },
            ('SDP', 'None'): {
                'name': 'Reach-SDP~\cite{hu2020reach}',
                'color': 'tab:red',
                'ls': '--',
            },
            ('SeparableCROWN', 'None'): {
                'name': 'CL-CROWN',
            },
            ('SeparableSGIBP', 'None'): {
                'name': 'CL-SG-IBP~\cite{xiang2020reachable}',
            },
            ('randUP', 'randUP'): {
                'name': 'RandUP',
                'color': 'k',
                'ls': '-',
            },
            ('randUP_M1k', 'randUP_M1k'): {
                'name': 'RandUP_M1k',
                'color': 'k',
                'ls': '-',
            },
            ('randUP_M10k', 'randUP_M10k'): {
                'name': 'RandUP_M10k',
                'color': 'k',
                'ls': '-',
            },
            ('kernelUP_M100', 'kernelUP_M100'): {
                'name': 'kernelUP_M100',
                'color': 'tab:blue',
                'ls': '-',
            },
            ('kernelUP_M300', 'kernelUP_M300'): {
                'name': 'kernelUP_M300',
                'color': 'tab:blue',
                'ls': '-',
            },
            ('kernelUP_M1k', 'kernelUP_M1k'): {
                'name': 'kernelUP_M1k',
                'color': 'tab:blue',
                'ls': '-',
            },
            ('GoTube_M1k', 'GoTube_M1k'): {
                'name': 'GoTube_M1k',
                'color': 'tab:red',
                'ls': '-',
            },
            ('GoTube_M3k', 'GoTube_M3k'): {
                'name': 'GoTube_M3k',
                'color': 'tab:red',
                'ls': '-',
            },
            ('GoTube_M10k', 'GoTube_M10k'): {
                'name': 'GoTube_M10k',
                'color': 'tab:red',
                'ls': '-',
            },
        }

class NNVerifExperiment(Experiment):
    def __init__(self, system="double_integrator", 
                       controller_model="double_integrator",
                       filename=""):
        if filename == "":
            self.filename = results_dir + 'exp_{dt}.pkl'
        else:
            self.filename = filename
        self.system = system
        self.controller_model = controller_model
        Experiment.__init__(self)

    def run(self, expts):
        dt = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

        parser = ex.setup_parser()
        args = parser.parse_args()

        args.save_plot = False
        args.show_plot = False
        args.make_animation = False
        args.show_animation = False
        args.init_state_range = "[[2.5, 3.0], [-0.1, 0.1]]"
        args.state_feedback = True
        args.system = self.system
        args.controller_model = self.controller_model
        args.t_max = 4 # 9 
        args.estimate_runtime = True
        args.num_calls = 2#100


        df = pd.DataFrame()

        for expt in expts:
            for key, value in expt.items():
                setattr(args, key, value)
            if args.sampling_based:
                args.boundaries = "lp"
            else:
                if expt['boundaries']=='lp':
                    args.boundaries = 'lp'
                elif expt['boundaries']=='polytope':
                    args.boundaries = 'polytope'
                else:
                    raise NotImplementedError("Unimplemented boundary type.")

            stats, info = ex.main(args)

            nb_samples = 0
            eps_pad    = 0.
            Lambda     = 0.
            sigma      = 0.
            if args.sampling_based:
                nb_samples = args.nb_samples
                if args.randUP or args.GoTube:
                    eps_pad = args.epsilon
                elif args.kernelUP:
                    Lambda = args.Lambda
                    sigma  = args.sigma

            for i, runtime in enumerate(stats['runtimes']):
                df = df.append({
                    **expt,
                    'run': i,
                    'runtime': runtime,
                    'output_constraint': stats['output_constraints'][i],
                    'area_final_step_error': stats['area_final_step_errors'][i],
                    'area_avg_error': stats['area_avg_errors'][i],
                    'area_all_errors': stats['area_all_errors'][i],
                    'haus_final_step_error': stats['haus_final_step_errors'][i],
                    'haus_avg_error': stats['haus_avg_errors'][i],
                    'haus_all_errors': stats['haus_all_errors'][i],
                    'B_all_conserv': stats['B_all_conserv'][i],
                    'B_vec_conserv': stats['B_vec_conserv'][i],
                    'nb_samples': nb_samples,
                    'eps_pad': eps_pad,
                    'Lambda': Lambda,
                    'sigma': sigma,
                    }, ignore_index=True)
        df.to_pickle(self.filename.format(dt=dt))

    def grab_latest_groups(self):
        # Grab latest file as pandas dataframe
        list_of_files = glob.glob(self.filename.format(dt='*'))
        latest_filename = max(list_of_files, key=os.path.getctime)
        df = pd.read_pickle(latest_filename)

        # df will have every trial, so group by which prop/part was used
        groupby = ['propagator', 'partitioner', 'boundaries']
        grouped = df.groupby(groupby)
        return grouped, latest_filename

    def print_latex_table(self):
        grouped, filename = self.grab_latest_groups()

        # Setup table columns
        rows = []
        # rows.append(["Algorithm", "Runtime [s]", "Error"])
        rows.append(["Algorithm", "Runtime [ms]", "Area Error [%]", "Hausd. Dist", "Conserv. [%]"])

        tuples = []
        # tuples += [('SeparableCROWN', 'None'), ('SeparableSGIBP', 'None')]
        tuples += [(prop, part, boundaries) 
                    for part in ['None', 'Uniform'] 
                    for prop in ['SDP', 'CROWN']
                    for boundaries in ['lp', 'polytope']]
        tuples += [('randUP', 'randUP', 'hull'),]
        tuples += [('randUP_M1k', 'randUP_M1k', 'hull'),]
        tuples += [('randUP_M10k', 'randUP_M10k', 'hull'),]
        tuples += [('kernelUP_M100', 'kernelUP_M100', 'kernel'),]
        tuples += [('kernelUP_M300', 'kernelUP_M300', 'kernel'),]
        tuples += [('kernelUP_M1k', 'kernelUP_M1k', 'kernel'),]
        tuples += [('GoTube_M1k', 'GoTube_M1k', 'ball'),]
        tuples += [('GoTube_M3k', 'GoTube_M3k', 'ball'),]
        tuples += [('GoTube_M10k', 'GoTube_M10k', 'ball'),]

        # Go through each combination of prop/part we want in the table
        for prop_part_tuple in tuples:
            try:
                group = grouped.get_group(prop_part_tuple)
            except KeyError:
                continue
            prop_part_tuple = prop_part_tuple[:2] # no need for boundary type

            name = self.info[prop_part_tuple]['name']

            mean_runtime = group['runtime'].mean()
            std_runtime = group['runtime'].std()
            mean_runtime *= 1000.
            std_runtime *= 1000.
            runtime_str = "${:.3f} \pm {:.3f}$".format(mean_runtime, std_runtime)

            area_final_step_error = group['area_final_step_error'].mean()
            area_final_step_error *= 100.
            haus_final_step_error = group['haus_final_step_error'].mean()
            
            percent_conservative = group['B_all_conserv'].mean()
            percent_conservative *= 100.

            # Add the entries to the table for that prop/part
            row = []
            row.append(name)
            row.append(runtime_str)
            row.append(round(area_final_step_error))
            row.append(round(haus_final_step_error,4))
            row.append(round(percent_conservative,3))

            rows.append(row)

        # print as a human-readable table and as a latex table
        print(tabulate(rows, headers='firstrow'))
        print()
        print(tabulate(rows, headers='firstrow', tablefmt='latex_raw'))

    def plot_error_vs_timestep(self):
        grouped, filename = self.grab_latest_groups()

        fig, ax = plt.subplots(1, 1)

        # Go through each combination of prop/part we want in the table
        for propagator in ['SDP', 'CROWN']:
            for partitioner in ['None', 'Uniform']:
                for boundaries in ['lp, polytope']:
                    prop_part_tuple = (propagator, partitioner, boundaries)
                    try:
                        group = grouped.get_group(prop_part_tuple)
                    except KeyError:
                        continue

                    all_errors = group['area_all_errors'].iloc[0]
                    t_max = all_errors.shape[0]
                    label = self.info[prop_part_tuple]['name']

                    # replace citation with the ref number in this plot
                    label = label.replace('~\\cite{hu2020reach}', ' [22]')
                    
                    plt.plot(
                        np.arange(1, t_max+1),
                        all_errors, 
                        color=self.info[prop_part_tuple]['color'],
                        ls=self.info[prop_part_tuple]['ls'],
                        label=label,
                    )
        plt.legend()

        ax.set_yscale('log')
        plt.xlabel('Time Steps')
        plt.ylabel('Approximation Error')
        plt.tight_layout()

        # Save plot with similar name to pkl file that contains data
        fig_filename = filename.replace('table', 'timestep').replace('pkl', 'png')
        plt.savefig(fig_filename)

        # plt.show()

    def plot_reachable_sets(self, system="double_integrator", 
                                  controller_model="double_integrator"):

        grouped, filename = self.grab_latest_groups()

        if system=="double_integrator":
            dyn = dynamics.DoubleIntegrator()
            controller = load_controller(name=controller_model)
        else:
            raise NotImplementedError("Plotting not implemented for this system.")

        init_state_range = np.array(
            [  # (num_inputs, 2)
                [2.5, 3.0],  # x0min, x0max
                [-0.1, 0.1],  # x1min, x1max
            ]
        )

        partitioner_hyperparams = {
            "type": "None",
        }
        propagator_hyperparams = {
            "type": "CROWN",
            "input_shape": init_state_range.shape[:-1],
        }

        # Set up analyzer (+ parititoner + propagator)
        analyzer = analyzers.ClosedLoopAnalyzer(controller, dyn)
        analyzer.partitioner = partitioner_hyperparams
        analyzer.propagator = propagator_hyperparams

        input_constraint = constraints.LpConstraint(
            range=init_state_range, p=np.inf
        )

        inputs_to_highlight = [
            {"dim": [0], "name": "$\mathbf{x}_0$"},
            {"dim": [1], "name": "$\mathbf{x}_1$"},
        ]

        t_max = 4

        analyzer.partitioner.setup_visualization(
            input_constraint,
            t_max,
            analyzer.propagator,
            show_samples=True,
            show_samples_labels=True,
            inputs_to_highlight=inputs_to_highlight,
            aspect="auto",
            initial_set_color=analyzer.initial_set_color,
            initial_set_zorder=analyzer.initial_set_zorder,
            sample_zorder=analyzer.sample_zorder,
        )

        analyzer.partitioner.linewidth = 1

        # Plot initial set text
        # ax = analyzer.partitioner.animate_axes
        # x = (init_state_range[0,0]+init_state_range[0,1])/2.0 - 0.2
        # y = init_state_range[1,1] + 0.05
        # ax.text(x, y, r'$\mathcal{X}_0$', color='k', fontsize=16)

        # Go through each combination of prop/part we want in the table
        for propagator in ['SDP','CROWN',
                           'randUP','randUP_M1k','randUP_M10k',
                           'kernelUP_M100','kernelUP_M300','kernelUP_M1k',
                           'GoTube_M1k']:
            for partitioner in ['None','Uniform',
                                'randUP','randUP_M1k','randUP_M10k',
                                'kernelUP_M100','kernelUP_M300','kernelUP_M1k',
                                'GoTube_M1k']:
                for boundaries in ['lp, polytope','hull','kernel','ball']:
                    prop_part_tuple = (propagator, partitioner, boundaries)
                    try:
                        group = grouped.get_group(prop_part_tuple)
                    except KeyError:
                        continue

                    input_dims = [[0], [1]]
                    output_constraint = group['output_constraint'].iloc[0]

                    if (propagator=='randUP' and partitioner=='randUP') or \
                       (propagator=='randUP_M1k' and partitioner=='randUP_M1k') or \
                       (propagator=='randUP_M10k' and partitioner=='randUP_M10k'):
                        prop_part_tuple = prop_part_tuple[:2] # no need for boundaries type
                        nb_samples = group['nb_samples'].iloc[0]
                        eps_pad = group['eps_pad'].iloc[0]
                        UP = randUP(dyn, 
                                    controller, 
                                    nb_samples=nb_samples,
                                    padding_eps=eps_pad)
                        UP.visualize(
                            analyzer.partitioner.animate_axes,
                            input_dims,
                            output_constraint,
                            None,
                            reachable_set_color=self.info[prop_part_tuple]['color'],
                            reachable_set_ls=self.info[prop_part_tuple]['ls'],
                            reachable_set_zorder=0,
                            B_show_label=True
                        )
                    elif (propagator=='kernelUP_M100' and partitioner=='kernelUP_M100') or \
                         (propagator=='kernelUP_M300' and partitioner=='kernelUP_M300') or \
                         (propagator=='kernelUP_M1k' and partitioner=='kernelUP_M1k'):
                        prop_part_tuple = prop_part_tuple[:2] # no need for boundaries type
                        nb_samples = group['nb_samples'].iloc[0]
                        Lambda = group['Lambda'].iloc[0]
                        sigma = group['sigma'].iloc[0]
                        UP = kernelUP(dyn, 
                                      controller, 
                                      nb_samples=nb_samples,
                                      Lambda=Lambda,
                                      sigma=sigma)
                        UP.visualize(
                            analyzer.partitioner.animate_axes,
                            input_dims,
                            output_constraint,
                            None,
                            reachable_set_color=self.info[prop_part_tuple]['color'],
                            reachable_set_ls=self.info[prop_part_tuple]['ls'],
                            reachable_set_zorder=0,
                            B_show_label=True
                        )
                    elif (propagator=='GoTube_M1k' and partitioner=='GoTube_M1k') or \
                         (propagator=='GoTube_M3k' and partitioner=='GoTube_M3k') or \
                         (propagator=='GoTube_M10k' and partitioner=='GoTube_M10k'):
                        prop_part_tuple = prop_part_tuple[:2] # no need for boundaries type
                        nb_samples = group['nb_samples'].iloc[0]
                        eps_pad = group['eps_pad'].iloc[0]
                        UP = GoTube(dyn, 
                                    controller, 
                                    nb_samples=nb_samples,
                                    padding_eps=eps_pad)
                        UP.visualize(
                            analyzer.partitioner.animate_axes,
                            input_dims,
                            output_constraint,
                            None,
                            reachable_set_color=self.info[prop_part_tuple]['color'],
                            reachable_set_ls=self.info[prop_part_tuple]['ls'],
                            reachable_set_zorder=0,
                            B_show_label=True
                        )

                    else:
                        print("info[prop_part_tuple]['ls']=",self.info[prop_part_tuple]['ls'])
                        analyzer.partitioner.visualize(
                            [],
                            [],
                            output_constraint,
                            None,
                            reachable_set_color=self.info[prop_part_tuple]['color'],
                            reachable_set_ls=self.info[prop_part_tuple]['ls'],
                            reachable_set_zorder=analyzer.reachable_set_zorder,
                            B_show_label=True
                        )

                        analyzer.partitioner.default_patches = analyzer.partitioner.animate_axes.patches.copy()
                        analyzer.partitioner.default_lines = analyzer.partitioner.animate_axes.lines.copy()

        plt.legend()
        plt.tight_layout()

        # Save plot with similar name to pkl file that contains data
        fig_filename = filename.replace('table', 'reachable').replace('pkl', 'png')
        plt.savefig(fig_filename)

    def plot_error_vs_time(self):
        df = pd.read_pickle(self.filename)

        # --------------------------------------------------
        # RandUP
        M_randUP_vec = []
        dH_randUP_vec = []
        compTime_randUP_vec = []
        groupby = ['randUP', 'nb_samples']
        grouped = df.groupby(groupby)
        for exp in expts:
            if exp['sampling_based'] and exp['randUP']:
                M_randUP_vec.append(exp['nb_samples'])
        # Go through data structure
        tuples = [(True, M) for M in M_randUP_vec]
        for prop_part_tuple in tuples:
            try:
                group = grouped.get_group(prop_part_tuple)
            except KeyError:
                continue
            dH_randUP_vec.append(group['haus_final_step_error'].mean())
            compTime_randUP_vec.append(group['runtime'].mean())
        # --------------------------------------------------

        # --------------------------------------------------
        # kernel
        M_kernel_vec = []
        dH_kernel_vec = []
        compTime_kernel_vec = []
        groupby = ['kernelUP', 'nb_samples']
        grouped = df.groupby(groupby)
        for exp in expts:
            if exp['sampling_based'] and exp['kernelUP']:
                M_kernel_vec.append(exp['nb_samples'])
        # Go through data structure
        tuples = [(True, M) for M in M_kernel_vec]
        for prop_part_tuple in tuples:
            try:
                group = grouped.get_group(prop_part_tuple)
            except KeyError:
                continue
            dH_kernel_vec.append(group['haus_final_step_error'].mean())
            compTime_kernel_vec.append(group['runtime'].mean())
        # --------------------------------------------------

        # --------------------------------------------------
        # GoTube
        M_GoTube_vec = []
        dH_GoTube_vec = []
        compTime_GoTube_vec = []
        groupby = ['GoTube', 'nb_samples']
        grouped = df.groupby(groupby)
        for exp in expts:
            if exp['sampling_based'] and exp['GoTube']:
                M_GoTube_vec.append(exp['nb_samples'])
        # Go through data structure
        tuples = [(True, M) for M in M_GoTube_vec]
        for prop_part_tuple in tuples:
            try:
                group = grouped.get_group(prop_part_tuple)
            except KeyError:
                continue
            dH_GoTube_vec.append(group['haus_final_step_error'].mean())
            compTime_GoTube_vec.append(group['runtime'].mean())
        # --------------------------------------------------

        # --------------------------------------------------
        # Formal Method
        dH_ReachLP_noSplit_lp         = 0
        dH_ReachLP_noSplit_poly       = 0
        dH_ReachLP_Split_lp           = 0
        dH_ReachLP_Split_poly         = 0
        compTime_ReachLP_noSplit_lp   = 0
        compTime_ReachLP_noSplit_poly = 0
        compTime_ReachLP_Split_lp     = 0
        compTime_ReachLP_Split_poly   = 0
        groupby = ['propagator', 'partitioner', 'boundaries']
        grouped = df.groupby(groupby)
        for propagator in ['SDP', 'CROWN']:
            for partitioner in ['None', 'Uniform']:
                for boundary in ['lp', 'polytope']:
                    prop_part_tuple = (propagator, partitioner, boundary)
                    try:
                        group = grouped.get_group(prop_part_tuple)
                    except KeyError:
                        continue
                    if propagator=='CROWN' and partitioner=='None':
                        if boundary=='lp':
                            compTime_ReachLP_noSplit_lp = group['runtime'].mean()
                            dH_ReachLP_noSplit_lp       = group['haus_final_step_error'].mean()
                        if boundary=='polytope':
                            compTime_ReachLP_noSplit_poly = group['runtime'].mean()
                            dH_ReachLP_noSplit_poly       = group['haus_final_step_error'].mean()
                    if propagator=='CROWN' and partitioner=='Uniform':
                        if boundary=='lp':
                            compTime_ReachLP_Split_lp = group['runtime'].mean()
                            dH_ReachLP_Split_lp       = group['haus_final_step_error'].mean()
                        if boundary=='polytope':
                            compTime_ReachLP_Split_poly = group['runtime'].mean()
                            dH_ReachLP_Split_poly       = group['haus_final_step_error'].mean()

        # --------------------------------------------------

        fig = plt.figure(figsize=(6.4,4.2))
        # plt.plot(M_randUP_vec, dH_randUP_vec, c='crimson',   label=r'RandUP', linewidth=2)
        # plt.plot(M_kernel_vec, dH_kernel_vec, c='chocolate', label=r'Kernel', linewidth=2)
        # plt.plot(M_GoTube_vec, dH_GoTube_vec, c='darkgreen', label=r'GoTube', linewidth=2)
        plt.plot(M_randUP_vec, dH_randUP_vec, c='tab:orange', label=r'RandUP', linewidth=2)
        plt.plot(M_kernel_vec, dH_kernel_vec, c='tab:blue',   label=r'Kernel', linewidth=2)
        plt.plot(M_GoTube_vec, dH_GoTube_vec, c='tab:red',    label=r'GoTube', linewidth=2)
        plt.plot(M_randUP_vec, dH_ReachLP_noSplit_lp*np.ones(len(M_randUP_vec)), 
                    c='b', linestyle=(0, (3,5,1,5)), linewidth=2)
        plt.plot(M_randUP_vec, dH_ReachLP_noSplit_poly*np.ones(len(M_randUP_vec)), 
                    c='b', linestyle=(0, (5,5)), linewidth=2)
        plt.plot(M_randUP_vec, dH_ReachLP_Split_lp*np.ones(len(M_randUP_vec)), 
                    c='c', linestyle=(0, (3,5,1,5)), linewidth=2)
        plt.plot(M_randUP_vec, dH_ReachLP_Split_poly*np.ones(len(M_randUP_vec)), 
                    c='c', linestyle=(0, (5,5)), linewidth=2)

        plt.text(M_randUP_vec[0]-3, dH_ReachLP_noSplit_poly+0.08, r'ReachLP NoSplit', 
                    color='b', fontsize=20)
        plt.text(1475, dH_ReachLP_noSplit_lp+0.1, r'rectangular output', 
                    color='b', fontsize=18)
        plt.text(1800, dH_ReachLP_noSplit_poly+0.04, r'polytopic output', 
                    color='b', fontsize=18)

        plt.text(M_randUP_vec[0]-3, dH_ReachLP_Split_poly+0.01, r'ReachLP Split', 
                    color='c', fontsize=20)
        plt.text(1475, dH_ReachLP_Split_lp+0.01, r'rectangular output', 
                    color='c', fontsize=18)
        plt.text(1800, dH_ReachLP_Split_poly+0.004, r'polytopic output', 
                    color='c', fontsize=18)

        plt.xlabel(r'$M$',                fontsize=30)
        plt.ylabel(r'$d_H(\hat{\mathcal{Y}}^M,\mathcal{Y}\,)$', fontsize=30)
        # plt.ylabel(r'$d_H(\hat{\mathcal{X}}_4,\mathcal{X}_4)$', fontsize=30)
        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)
        # plt.gca().axes.xaxis.set_ticklabels([])
        plt.gca().set_xscale('log')
        plt.gca().set_yscale('log')
        plt.grid(which='minor', alpha=0.5, linestyle='--')
        plt.grid(which='major', alpha=0.75, linestyle=':')
        plt.subplots_adjust(left=0.21, bottom=0.105, right=0.962, top=0.975, wspace=0.2, hspace=0.2)
        plt.ylim(7e-4,1.3)
        plt.legend(fontsize=22, loc='lower left', #bbox_to_anchor=(0.1, 0.3), 
                    labelspacing=0.1, handlelength=1., handletextpad=0.3, borderaxespad=0.2, framealpha=1)
        plt.savefig(self.filename+"_dH.png")
        # plt.show()

        # --------------------------------------------------

        fig = plt.figure(figsize=(6.4,4))
        plt.plot(M_randUP_vec, compTime_randUP_vec, c='tab:orange', linewidth=2)
        plt.plot(M_kernel_vec, compTime_kernel_vec, c='tab:blue',   linewidth=2)
        plt.plot(M_GoTube_vec, compTime_GoTube_vec, c='tab:red',    linewidth=2)

        plt.plot(M_randUP_vec, compTime_ReachLP_noSplit_lp*np.ones(len(M_randUP_vec)), 
                    c='b', linestyle=(0, (3,5,1,5)), linewidth=2)
        plt.plot(M_randUP_vec, compTime_ReachLP_noSplit_poly*np.ones(len(M_randUP_vec)), 
                    c='b', linestyle=(0, (5,5)), linewidth=2)
        plt.text(M_randUP_vec[0]-3, compTime_ReachLP_noSplit_lp+0.035, r'ReachLP', 
                    color='b', fontsize=22)
        plt.text(M_randUP_vec[0]-3, compTime_ReachLP_noSplit_lp+0.01, r'NoSplit', 
                    color='b', fontsize=22)
        plt.text(1475, compTime_ReachLP_noSplit_lp+0.003, r'rectangular output', 
                    color='b', fontsize=18)
        plt.text(1800, compTime_ReachLP_noSplit_poly-0.06, r'polytopic output', 
                    color='b', fontsize=18)

        plt.plot(M_randUP_vec, compTime_ReachLP_Split_lp*np.ones(len(M_randUP_vec)), 
                    c='c', linestyle=(0, (3,5,1,5)), linewidth=2)
        plt.plot(M_randUP_vec, compTime_ReachLP_Split_poly*np.ones(len(M_randUP_vec)), 
                    c='c', linestyle=(0, (5,5)), linewidth=2)
        plt.text(M_randUP_vec[0]-3, compTime_ReachLP_Split_lp+0.5, r'ReachLP', 
                    color='c', fontsize=22)
        plt.text(M_randUP_vec[0]-3, compTime_ReachLP_Split_lp+0.15, r'Split', 
                    color='c', fontsize=22)
        plt.text(1475, compTime_ReachLP_Split_lp+0.05, r'rectangular output', 
                    color='c', fontsize=18)
        plt.text(1800, compTime_ReachLP_Split_poly-0.9, r'polytopic output', 
                    color='c', fontsize=18)

        plt.xlabel(r'$M$',       fontsize=30)
        plt.ylabel(r'Time [s]', fontsize=30)
        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)
        plt.gca().set_xscale('log')
        plt.gca().set_yscale('log')
        plt.grid(which='minor', alpha=0.5, linestyle='--')
        plt.grid(which='major', alpha=0.75, linestyle=':')
        plt.ylim(0.0009, 3.)
        plt.subplots_adjust(left=0.21, bottom=0.205, right=0.962, top=0.99, wspace=0.2, hspace=0.2)
        plt.savefig(self.filename+"_comp_time.png")
        # plt.show()

        # --------------------------------------------------


if __name__ == '__main__':
    system_name = "double_integrator"
    controller_model_name = system_name

    B_compute      = True
    B_plot         = False
    B_get_tab      = False
    B_plot_M_vs_dH = True

    filename = results_dir+"hausdorff_dist_plots.pkl"
    if B_compute:
        c = NNVerifExperiment(system=system_name, 
                              controller_model=controller_model_name,
                              filename=filename)
        c.run(expts)

    c = NNVerifExperiment(system=system_name, 
                          controller_model=controller_model_name,
                          filename=filename)
    if B_plot:
        print("Reachable sets are plotted with epsilon=0 for RandUP and GoTube.")
        c.plot_reachable_sets(system=system_name, 
                              controller_model=controller_model_name)

    if B_get_tab:
        c.print_latex_table()

    if B_plot_M_vs_dH:
        c.plot_error_vs_time()