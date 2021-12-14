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
    #
    #
    #
    #
    #
    {
        'partitioner': 'randUP_M1k',
        'propagator': 'randUP_M1k',
        'sampling_based': True,
        'boundaries': 'hull',
        'randUP': True,
        'kernelUP': False,
        'GoTube': False,
        'nb_samples': 1000,
        'epsilon': 0.02,
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
        'partitioner': 'GoTube_M1k',
        'propagator': 'GoTube_M1k',
        'sampling_based': True,
        'boundaries': 'ball',
        'randUP': False,
        'kernelUP': False,
        'GoTube': True,
        'nb_samples': 1000,
        'epsilon': 0.02,
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

    def run(self, expts, t_max=9):
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
        args.t_max = t_max
        args.estimate_runtime = True

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

    def plot_reachable_sets(self, system="double_integrator", 
                                  controller_model="double_integrator",
                                  t_max=9,
                                  B_plot_formal_lp=True,
                                  B_plot_formal_poly=True,
                                  B_zoom=True):
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
                           'randUP','randUP_M1k','randUP_M10k',
                           'kernelUP_M100','kernelUP_M300','kernelUP_M1k',
                           'GoTube_M1k']:
            for partitioner in ['None','Uniform',
                                'randUP','randUP_M1k','randUP_M10k',
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
                            B_show_label=True,
                            t_max=t_max
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

                        analyzer.partitioner.default_patches = analyzer.partitioner.animate_axes.patches.copy()
                        analyzer.partitioner.default_lines = analyzer.partitioner.animate_axes.lines.copy()

        # plt.legend(fontsize=20)
        plt.legend(fontsize=28, loc='upper right', #bbox_to_anchor=(0.1, 0.3), 
                    labelspacing=0.1, handlelength=1., handletextpad=0.3, borderaxespad=0.2, framealpha=1)
        # plt.xlim([-1.10,3.1])
        # plt.ylim([-1.35,0.3])
        plt.tight_layout()

        if B_zoom:
            plt.xlim([-0.015,0.35  ])
            plt.ylim([-0.375,-0.1])
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



if __name__ == '__main__':
    system_name = "double_integrator"
    controller_model_name = system_name

    B_compute      = True
    B_plot         = True
    B_get_tab      = False

    B_plot_formal_lp   = True
    B_plot_formal_poly = False

    B_zoom = False

    filename = results_dir+"reachable_set_plots.pkl"
    if B_compute:
        c = NNVerifExperiment(system=system_name, 
                              controller_model=controller_model_name,
                              filename=filename)
        c.run(expts, t_max=5)

    # Plotting, change filename to latest
    c = NNVerifExperiment(system=system_name, 
                          controller_model=controller_model_name,
                          filename=filename)
    if B_plot:
        c.plot_reachable_sets(system=system_name, 
                              controller_model=controller_model_name,
                              t_max=5,
                              B_plot_formal_lp=B_plot_formal_lp,
                              B_plot_formal_poly=B_plot_formal_poly,
                              B_zoom=B_zoom)

    if B_get_tab:
        c.print_latex_table()