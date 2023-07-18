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
import nn_closed_loop.sampling_based.pmpUP as pmpUP
import nn_closed_loop.sampling_based.kernelUP as kernelUP
import nn_closed_loop.sampling_based.GoTube as GoTube
from nn_closed_loop.utils.nn import load_controller, replace_relus_to_softplus


results_dir = "{}/results/logs/".format(
    os.path.dirname(os.path.abspath(__file__))
)
os.makedirs(results_dir, exist_ok=True)


expts = [
    {
        'partitioner': 'None',
        'propagator': 'CROWN',
        'sampling_based': False,
        'boundaries': 'lp'
    },
    {
        'partitioner': 'randUP_M100',
        'propagator': 'randUP_M100',
        'sampling_based': True,
        'boundaries': 'hull',
        'randUP': True,
        'pmpUP': False,
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
        'pmpUP': False,
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
        'pmpUP': False,
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
        'pmpUP': False,
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
        'pmpUP': False,
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
        'pmpUP': False,
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
        'pmpUP': False,
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
        'pmpUP': False,
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
        'pmpUP': False,
        'kernelUP': False,
        'GoTube': False,
        'nb_samples': 10000,
        'epsilon': 0.0,
    },
    {
        'partitioner': 'pmpUP_M50',
        'propagator': 'pmpUP_M50',
        'sampling_based': True,
        'boundaries': 'hull',
        'randUP': False,
        'pmpUP': True,
        'kernelUP': False,
        'GoTube': False,
        'nb_samples': 50,
        'epsilon': 0.0,
    },
    {
        'partitioner': 'pmpUP_M100',
        'propagator': 'pmpUP_M100',
        'sampling_based': True,
        'boundaries': 'hull',
        'randUP': False,
        'pmpUP': True,
        'kernelUP': False,
        'GoTube': False,
        'nb_samples': 100,
        'epsilon': 0.0,
    },
    {
        'partitioner': 'pmpUP_M200',
        'propagator': 'pmpUP_M200',
        'sampling_based': True,
        'boundaries': 'hull',
        'randUP': False,
        'pmpUP': True,
        'kernelUP': False,
        'GoTube': False,
        'nb_samples': 200,
        'epsilon': 0.0,
    },
    {
        'partitioner': 'pmpUP_M300',
        'propagator': 'pmpUP_M300',
        'sampling_based': True,
        'boundaries': 'hull',
        'randUP': False,
        'pmpUP': True,
        'kernelUP': False,
        'GoTube': False,
        'nb_samples': 300,
        'epsilon': 0.0,
    },
    {
        'partitioner': 'pmpUP_M500',
        'propagator': 'pmpUP_M500',
        'sampling_based': True,
        'boundaries': 'hull',
        'randUP': False,
        'pmpUP': True,
        'kernelUP': False,
        'GoTube': False,
        'nb_samples': 500,
        'epsilon': 0.0,
    },
    {
        'partitioner': 'pmpUP_M1k',
        'propagator': 'pmpUP_M1k',
        'sampling_based': True,
        'boundaries': 'hull',
        'randUP': False,
        'pmpUP': True,
        'kernelUP': False,
        'GoTube': False,
        'nb_samples': 1000,
        'epsilon': 0.0,
    },
    {
        'partitioner': 'pmpUP_M2k',
        'propagator': 'pmpUP_M2k',
        'sampling_based': True,
        'boundaries': 'hull',
        'randUP': False,
        'pmpUP': True,
        'kernelUP': False,
        'GoTube': False,
        'nb_samples': 2000,
        'epsilon': 0.0,
    },
    {
        'partitioner': 'pmpUP_M5k',
        'propagator': 'pmpUP_M5k',
        'sampling_based': True,
        'boundaries': 'hull',
        'randUP': False,
        'pmpUP': True,
        'kernelUP': False,
        'GoTube': False,
        'nb_samples': 5000,
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
            ('randUP_M500', 'randUP_M500'): {
                'name': 'RandUP_M500',
                'color': 'k',
                'ls': '-',
            },
            ('randUP_M1k', 'randUP_M1k'): {
                'name': 'RandUP_M1k',
                'color': 'k',
                'ls': '-',
            },
            ('randUP_M2k', 'randUP_M2k'): {
                'name': 'RandUP_M2k',
                'color': 'k',
                'ls': '-',
            },
            ('randUP_M3k', 'randUP_M3k'): {
                'name': 'RandUP_M3k',
                'color': 'k',
                'ls': '-',
            },
            ('randUP_M5k', 'randUP_M5k'): {
                'name': 'RandUP_M5k',
                'color': 'k',
                'ls': '-',
            },
            ('randUP_M10k', 'randUP_M10k'): {
                'name': 'RandUP_M10k',
                'color': 'k',
                'ls': '-',
            },
            ('pmpUP_M200', 'pmpUP_M200'): {
                'name': 'pmpUP_M200',
                'color': 'k',
                'ls': '-',
            },
            ('pmpUP_M500', 'pmpUP_M500'): {
                'name': 'pmpUP_M500',
                'color': 'k',
                'ls': '-',
            },
            ('pmpUP_M1k', 'pmpUP_M1k'): {
                'name': 'pmpUP_M1k',
                'color': 'k',
                'ls': '-',
            },
            ('pmpUP_M2k', 'pmpUP_M2k'): {
                'name': 'pmpUP_M2k',
                'color': 'k',
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
        args.init_state_range = "[[2.74999, 2.75001], [-1e-6, 1e-6]]"
        args.state_feedback = True
        args.system = self.system
        args.controller_model = self.controller_model
        args.t_max = 16 
        args.estimate_runtime = True
        args.num_calls = 5


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
                if args.randUP or args.pmpUP or args.GoTube:
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
        tuples += [('pmpUP_M1k', 'pmpUP_M1k', 'hull'),]
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

    def plot_reachable_sets(self, system="double_integrator", 
                                  controller_model="double_integrator",
                                  t_max=16,
                                  B_plot_formal_lp=True,
                                  B_plot_formal_poly=True,
                                  B_zoom=False):
        grouped, filename = self.grab_latest_groups()

        if system=="double_integrator":
            dyn = dynamics.DoubleIntegrator()
            controller = load_controller(name=controller_model)
            replace_relus_to_softplus(controller)
        else:
            raise NotImplementedError("Plotting not implemented for this system.")

        init_state_range = np.array(
            [  # (num_inputs, 2)
                [2.74999, 2.75001],  # x0min, x0max
                [-1e-6, 1e-6],  # x1min, x1max
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
            {"dim": [0], "name": r"$x_0$"},
            {"dim": [1], "name": r"$x_1$"},
        ]

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
                           'randUP','randUP_M1k',#'randUP_M10k',
                           'pmpUP_M1k',
                           'kernelUP_M100','kernelUP_M300','kernelUP_M1k',
                           'GoTube_M1k']:
            for partitioner in ['None','Uniform',
                                'randUP','randUP_M1k',#'randUP_M10k',
                                'pmpUP_M1k',
                                'kernelUP_M100','kernelUP_M300','kernelUP_M1k',
                                'GoTube_M1k']:
                for boundaries in ['lp', 'polytope','hull','kernel','ball']:
                    prop_part_tuple = (propagator, partitioner, boundaries)
                    try:
                        group = grouped.get_group(prop_part_tuple)
                    except KeyError:
                        continue
                    print(prop_part_tuple)

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
                            B_show_label=True,
                            t_max=t_max
                        )

                    elif (propagator=='pmpUP' and partitioner=='pmpUP') or \
                       (propagator=='pmpUP_M1k' and partitioner=='pmpUP_M1k') or \
                       (propagator=='pmpUP_M10k' and partitioner=='pmpUP_M10k'):
                        prop_part_tuple = prop_part_tuple[:2] # no need for boundaries type
                        nb_samples = group['nb_samples'].iloc[0]
                        eps_pad = group['eps_pad'].iloc[0]
                        UP = pmpUP(dyn, 
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
                            B_show_label=True,
                            t_max=t_max
                        )

                    else:
                        prop_part_tuple = prop_part_tuple[:2] # no need for boundaries type
                        plot_label = "ReachLP"
                        # reduce to t_max
                        if isinstance(output_constraint, constraints.PolytopeConstraint):
                            plot_label = plot_label + ""
                            if not(B_plot_formal_poly):
                                continue
                            # output_constraint.A = output_constraint.A[:t_max] # A is the same for all of them
                            output_constraint.b = output_constraint.b[:t_max]
                        elif isinstance(output_constraint, constraints.LpConstraint):
                            if not(B_plot_formal_lp):
                                continue
                            output_constraint.range = output_constraint.range[:t_max]
                        else:
                            raise NotImplementedError
                        # plot
                        analyzer.partitioner.visualize(
                            [],
                            [],
                            output_constraint,
                            None,
                            reachable_set_color=self.info[prop_part_tuple]['color'],
                            reachable_set_ls=self.info[prop_part_tuple]['ls'],
                            reachable_set_zorder=analyzer.reachable_set_zorder,
                            B_show_label=True,
                            label=plot_label
                        )

                        # analyzer.partitioner.default_patches = analyzer.partitioner.animate_axes.patches.copy()
                        # analyzer.partitioner.default_lines = analyzer.partitioner.animate_axes.lines.copy()

        # plt.legend(fontsize=20)
        lgd = plt.legend(fontsize=34, #loc='upper center', 
            bbox_to_anchor=(0.41, 0.5, 0.5, 0.5), 
                    labelspacing=0.1, handlelength=1., handletextpad=0.3, borderaxespad=0.2, framealpha=1)
        plt.xlim([-0.15,2.8])
        plt.ylim([-0.885,0.02])
        plt.tight_layout()

        if B_zoom:
            plt.xlim([0.15,0.675])
            plt.ylim([-0.5, -0.25])
            frame = plt.gca()
            frame.get_legend().remove()
            frame.axes.xaxis.set_ticklabels([])
            frame.axes.yaxis.set_ticklabels([])
            frame.set_ylabel('')
            frame.set_xlabel('')
            # for line in frame.get_lines():
                # line.set_linewidth(19.)


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
        # pmpUP
        M_pmpUP_vec = []
        dH_pmpUP_vec = []
        compTime_pmpUP_vec = []
        groupby = ['pmpUP', 'nb_samples']
        grouped = df.groupby(groupby)
        for exp in expts:
            if exp['sampling_based'] and exp['pmpUP']:
                M_pmpUP_vec.append(exp['nb_samples'])
        # Go through data structure
        tuples = [(True, M) for M in M_pmpUP_vec]
        for prop_part_tuple in tuples:
            try:
                group = grouped.get_group(prop_part_tuple)
            except KeyError:
                continue
            dH_pmpUP_vec.append(group['haus_final_step_error'].mean())
            compTime_pmpUP_vec.append(group['runtime'].mean())
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

        fig = plt.figure(figsize=(6.4,5))
        M_reachLP_vec = np.array([
            np.minimum(np.min(M_randUP_vec), np.min(M_pmpUP_vec)), 
            np.maximum(np.max(M_randUP_vec), np.max(M_pmpUP_vec))])
        plt.plot(M_reachLP_vec, dH_ReachLP_noSplit_lp*np.ones(2), 
                    c='tab:green', linestyle=(0, (3,3)), label=r'ReachLP', linewidth=3)
        plt.plot(M_randUP_vec, dH_randUP_vec, c='tab:orange', label=r'RandUP', linewidth=3)
        plt.plot(M_pmpUP_vec, dH_pmpUP_vec, c='tab:blue', label=r'Algorithm 1', linewidth=3)

        plt.legend(fontsize=30, loc='center right', #bbox_to_anchor=(0.1, 0.3), 
                    labelspacing=0.1, handlelength=1., handletextpad=0.3, borderaxespad=0.2, framealpha=1)
        plt.xlabel(r'Sample Size $M$', fontsize=30)
        plt.ylabel(r'Estimation Error', fontsize=30)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.gca().set_xscale('log')
        plt.gca().set_yscale('log')
        plt.grid(which='minor', alpha=0.5, linestyle='--')
        plt.grid(which='major', alpha=0.75, linestyle=':')
        plt.ylim(9e-5,1.3)
        plt.subplots_adjust(left=0.21, bottom=0.205, right=0.962, top=0.98, wspace=0.2, hspace=0.2)
        plt.savefig(self.filename+"_dH.png")
        # plt.show()

        # --------------------------------------------------

        fig = plt.figure(figsize=(6.4,5))
        plt.plot(M_randUP_vec, compTime_randUP_vec, c='tab:orange', label=r'RandUP', linewidth=2)
        plt.plot(M_pmpUP_vec, compTime_pmpUP_vec, c='tab:blue', label=r'Algorithm 1', linewidth=2)

        plt.xlabel(r'Num. samples $M$', fontsize=30)
        plt.ylabel(r'Time [s]', fontsize=30)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.gca().set_xscale('log')
        plt.gca().set_yscale('log')
        plt.grid(which='minor', alpha=0.5, linestyle='--')
        plt.grid(which='major', alpha=0.75, linestyle=':')
        plt.ylim(0.0009, 3.)
        plt.subplots_adjust(left=0.21, bottom=0.150, right=0.962, top=0.99, wspace=0.2, hspace=0.2)
        plt.legend(fontsize=30, loc='upper left', #bbox_to_anchor=(0.1, 0.3), 
            labelspacing=0.1, handlelength=1., handletextpad=0.3, borderaxespad=0.2, framealpha=1)
        plt.savefig(self.filename+"_comp_time.png")
        # plt.show()

        # --------------------------------------------------

        fig = plt.figure(figsize=(6.4,5))
        ax = plt.gca()
        print("compTime_ReachLP_noSplit_lp =", compTime_ReachLP_noSplit_lp)
        print("dH_ReachLP_noSplit_lp =", dH_ReachLP_noSplit_lp)
        plt.scatter(compTime_ReachLP_noSplit_lp, dH_ReachLP_noSplit_lp, 
                    c='tab:green', marker='+', label=r'ReachLP', linewidth=3, s=180)
        plt.plot(compTime_randUP_vec, dH_randUP_vec, c='tab:orange', label=r'RandUP', linewidth=3)
        plt.plot(compTime_pmpUP_vec, dH_pmpUP_vec, c='tab:blue', label=r'Algorithm 1', linewidth=3)
        plt.legend(fontsize=30, loc='center right', #bbox_to_anchor=(0.1, 0.3), 
                    labelspacing=0.1, handlelength=1., handletextpad=0.3, borderaxespad=0.2, framealpha=1)

        plt.xlabel(r'Time [s]',       fontsize=30)
        # plt.ylabel(r'$d_H(H(\mathcal{X}_T),\hat{H}(\mathcal{X}_T)\,)$', fontsize=30)
        plt.ylabel(r'Estimation Error', fontsize=30)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.grid(which='minor', alpha=0.5, linestyle='--')
        plt.grid(which='major', alpha=0.75, linestyle=':')
        plt.xlim(0.5e-2,1.4e-1)
        plt.ylim(9e-5,1.3)
        plt.subplots_adjust(left=0.21, bottom=0.205, right=0.962, top=0.98, wspace=0.2, hspace=0.2)
        plt.savefig(self.filename+"_dh_vs_comp_time.png")

        # --------------------------------------------------


if __name__ == '__main__':
    system_name = "double_integrator"
    controller_model_name = system_name

    B_compute      = True
    B_plot         = True
    B_get_tab      = True
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