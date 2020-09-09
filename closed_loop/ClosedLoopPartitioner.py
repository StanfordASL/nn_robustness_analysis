import numpy as np
from partition.Partitioner import Partitioner, UniformPartitioner
import pypoman
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
from closed_loop.nn import control_nn
from itertools import product
from closed_loop.utils import init_state_range_to_polytope


class ClosedLoopPartitioner(Partitioner):
    def __init__(self, At=None, bt=None, ct=None):
        Partitioner.__init__(self)
        self.At = At
        self.bt = bt
        self.ct = ct

    def get_one_step_reachable_set(self, A_inputs, b_inputs, A_out, propagator):
        reachable_set, info = propagator.get_one_step_reachable_set(A_inputs, b_inputs, A_out)
        return reachable_set, info

    def get_reachable_set(self, A_inputs, b_inputs, A_out, propagator, t_max):
        reachable_set, info = propagator.get_reachable_set(A_inputs, b_inputs, A_out, t_max)
        return reachable_set, info

    def setup_visualization(self, A_inputs, b_inputs, A_out, b_out, propagator, show_samples=True, outputs_to_highlight=None, inputs_to_highlight=None):
        self.animate_fig, self.animate_axes = plt.subplots(1,1)

        if inputs_to_highlight is None:
            # Automatically detect which input dims to show based on input_range
            # num_input_dimensions_to_plot = 2
            # input_shape = A_inputs.
            # lengths = input_range[...,1].flatten() - input_range[...,0].flatten()
            # flat_dims = np.argpartition(lengths, -num_input_dimensions_to_plot)[-num_input_dimensions_to_plot:]
            # flat_dims.sort()
            input_dims = [[0], [1]]
            # input_dims = [np.unravel_index(flat_dim, input_range.shape[:-1]) for flat_dim in flat_dims]
            input_names = ["State: {}".format(input_dims[0][0]), "State: {}".format(input_dims[1][0])]
        else:
            input_dims = [x['dim'] for x in inputs_to_highlight]
            input_names = [x['name'] for x in inputs_to_highlight]
        self.input_dims_ = tuple([tuple([input_dims[j][i] for j in range(len(input_dims))]) for i in range(len(input_dims[0]))])

        # scale = 0.05
        # x_off = max((input_range[input_dims[0]+(1,)] - input_range[input_dims[0]+(0,)])*(scale), 1e-5)
        # y_off = max((input_range[input_dims[1]+(1,)] - input_range[input_dims[1]+(0,)])*(scale), 1e-5)
        # self.animate_axes[0].set_xlim(input_range[input_dims[0]+(0,)] - x_off, input_range[input_dims[0]+(1,)]+x_off)
        # self.animate_axes[0].set_ylim(input_range[input_dims[1]+(0,)] - y_off, input_range[input_dims[1]+(1,)]+y_off)
        self.animate_axes.set_xlabel(input_names[0])
        self.animate_axes.set_ylabel(input_names[1])

        t_max = 5
        dt = 1.
        colors = [cm.get_cmap("tab10")(i) for i in range(t_max+1)]

        num_samples = 1000
        num_states = A_inputs.shape[-1]
        xs = np.zeros((num_samples, num_states))
        np.random.seed(0)
        dataset_index = 0

        while dataset_index < num_samples:

            # Initial state
            num_states = self.At.shape[0]
            x = np.zeros((int((t_max)/dt)+1, num_states))
            x[0,:] = np.random.uniform(
                low=[2.5,-0.25], 
                high=[3.0,0.25]
                # low=init_state_range[:,0], 
                # high=init_state_range[:,1]
            )
            this_colors = colors.copy()

            t = 0
            step = 0
            while t < t_max:
                t += dt
                u = control_nn(x=x[step,:], model=propagator.network, use_torch=True)
                # if clip_control:
                #     u = np.clip(u, u_min, u_max)
                # if collect_data:
                #     xs[dataset_index, :] = x[step,:]
                x[step+1,:] = np.dot(self.At, x[step, :]) + np.dot(self.bt,u)[:,0]
                step += 1
                dataset_index += 1
                if dataset_index == num_samples:
                    break

            self.animate_axes.scatter(x[:,0], x[:,1], c=this_colors)

        # # Make a rectangle for the Exact boundaries
        # sampled_outputs = self.get_sampled_outputs(input_range, propagator)
        # if show_samples:
        #     self.animate_axes.scatter(sampled_outputs[...,output_dims[0]], sampled_outputs[...,output_dims[1]], c='k', marker='.', zorder=2,
        #         label="Sampled States")

        # Initial state set
        # TODO: this doesn't use the computed input_dims...
        vertices = pypoman.compute_polygon_hull(A_inputs, b_inputs)
        bnd_color = 'k--'
        self.animate_axes.plot([v[0] for v in vertices]+[vertices[0][0]], [v[1] for v in vertices]+[vertices[0][1]],
            bnd_color, label='Initial States')

        # Reachable sets
        # TODO: this doesn't use the computed input_dims...
        for i in range(len(b_out)):
            vertices = pypoman.compute_polygon_hull(A_out, b_out[i])
            bnd_color = 'g'
            self.animate_axes.plot([v[0] for v in vertices]+[vertices[0][0]], [v[1] for v in vertices]+[vertices[0][1]],
                bnd_color, label='$\mathcal{R}_'+str(i+1)+'$')

        # self.default_patches = [[], []]
        # self.default_lines = [[], []]
        # self.default_patches[0] = [input_rect]
        
        # # Exact output range
        # color = 'black'
        # linewidth = 3
        # if self.interior_condition == "linf":
        #     output_range_exact = self.samples_to_range(sampled_outputs)
        #     output_range_exact_ = output_range_exact[self.output_dims_]
        #     rect = Rectangle(output_range_exact_[:2,0], output_range_exact_[0,1]-output_range_exact_[0,0], output_range_exact_[1,1]-output_range_exact_[1,0],
        #                     fc='none', linewidth=linewidth,edgecolor=color,
        #                     label="True Bounds ({})".format(label_dict[self.interior_condition]))
        #     self.animate_axes[1].add_patch(rect)
        #     self.default_patches[1].append(rect)
        # elif self.interior_condition == "lower_bnds":
        #     output_range_exact = self.samples_to_range(sampled_outputs)
        #     output_range_exact_ = output_range_exact[self.output_dims_]
        #     line1 = self.animate_axes[1].axhline(output_range_exact_[1,0], linewidth=linewidth,color=color,
        #         label="True Bounds ({})".format(label_dict[self.interior_condition]))
        #     line2 = self.animate_axes[1].axvline(output_range_exact_[0,0], linewidth=linewidth,color=color)
        #     self.default_lines[1].append(line1)
        #     self.default_lines[1].append(line2)
        # elif self.interior_condition == "convex_hull":
        #     from scipy.spatial import ConvexHull
        #     self.true_hull = ConvexHull(sampled_outputs)
        #     self.true_hull_ = ConvexHull(sampled_outputs[...,output_dims].squeeze())
        #     line = self.animate_axes[1].plot(
        #         np.append(sampled_outputs[self.true_hull_.vertices][...,output_dims[0]], sampled_outputs[self.true_hull_.vertices[0]][...,output_dims[0]]),
        #         np.append(sampled_outputs[self.true_hull_.vertices][...,output_dims[1]], sampled_outputs[self.true_hull_.vertices[0]][...,output_dims[1]]),
        #         color=color, linewidth=linewidth,
        #         label="True Bounds ({})".format(label_dict[self.interior_condition]))
        #     self.default_lines[1].append(line[0])
        # else:
        #     raise NotImplementedError

    def visualize(self, M, interior_M, A_out, b_out, iteration=0):
        # self.animate_axes.patches = self.default_patches[0].copy()
        # self.animate_axes.lines = self.default_lines[0].copy()
        input_dims_ = self.input_dims_

        # Rectangles that might still be outside the sim pts
        first = True
        for (input_range_, output_range_) in M:
            if first:
                input_label = 'Cell of Partition'
                output_label = "One Cell's Estimated Bounds"
                first = False
            else:
                input_label = None
                output_label = None

            for i in range(len(output_range_)):
                vertices = pypoman.compute_polygon_hull(A_out, output_range_[i])
                bnd_color = 'k'
                self.animate_axes.plot([v[0] for v in vertices]+[vertices[0][0]], [v[1] for v in vertices]+[vertices[0][1]],
                    bnd_color, label='$\mathcal{R}_'+str(i+1)+'$')

            rect = Rectangle(input_range_[:,0], input_range_[0,1]-input_range_[0,0], input_range_[1,1]-input_range_[1,0],
                    fc='none', linewidth=1,edgecolor='tab:purple')
            self.animate_axes.add_patch(rect)
            # vertices = pypoman.compute_polygon_hull(A_out, input_range[i])
            # bnd_color = 'k--'
            # self.animate_axes.plot([v[0] for v in vertices]+[vertices[0][0]], [v[1] for v in vertices]+[vertices[0][1]],
            #     bnd_color, label='$\mathcal{R}_'+str(i+1)+'$')

        # # Rectangles that are within the sim pts
        # for (input_range_, output_range_) in interior_M:
        #     output_range__ = output_range_[self.output_dims_]
        #     rect = Rectangle(output_range__[:2,0], output_range__[0,1]-output_range__[0,0], output_range__[1,1]-output_range__[1,0],
        #             fc='none', linewidth=1,edgecolor='tab:purple')
        #     self.animate_axes[1].add_patch(rect)

        #     input_range__ = input_range_[input_dims_]
        #     rect = Rectangle(input_range__[:,0], input_range__[0,1]-input_range__[0,0], input_range__[1,1]-input_range__[1,0],
        #             fc='none', linewidth=1,edgecolor='tab:purple')
        #     self.animate_axes[0].add_patch(rect)

        # linewidth = 2
        # color = 'tab:green'
        # if self.interior_condition == "linf":
        #     # Make a rectangle for the estimated boundaries
        #     output_range_estimate = self.squash_down_to_one_range(u_e, M)
        #     output_range_estimate_ = output_range_estimate[self.output_dims_]
        #     rect = Rectangle(output_range_estimate_[:2,0], output_range_estimate_[0,1]-output_range_estimate_[0,0], output_range_estimate_[1,1]-output_range_estimate_[1,0],
        #                     fc='none', linewidth=linewidth,edgecolor=color,
        #                     label="Estimated Bounds ({})".format(label_dict[self.interior_condition]))
        #     self.animate_axes[1].add_patch(rect)
        # elif self.interior_condition == "lower_bnds":
        #     output_range_estimate = self.squash_down_to_one_range(u_e, M)
        #     output_range_estimate_ = output_range_estimate[self.output_dims_]
        #     self.animate_axes[1].axhline(output_range_estimate_[1,0],
        #         linewidth=linewidth,color=color,
        #         label="Estimated Bounds ({})".format(label_dict[self.interior_condition]))
        #     self.animate_axes[1].axvline(output_range_estimate_[0,0],
        #         linewidth=linewidth,color=color)
        # elif self.interior_condition == "convex_hull":
        #     from scipy.spatial import ConvexHull
        #     M_ = [(input_range_, output_range_[self.output_dims_]) for (input_range_, output_range_) in M]
        #     hull = self.squash_down_to_convex_hull(M_, self.true_hull_.points)
        #     self.animate_axes[1].plot(
        #         np.append(hull.points[hull.vertices,0], hull.points[hull.vertices[0],0]),
        #         np.append(hull.points[hull.vertices,1], hull.points[hull.vertices[0],1]),
        #         color=color, linewidth=linewidth,
        #         label="Estimated Bounds ({})".format(label_dict[self.interior_condition]))
        # else:
        #     raise NotImplementedError

        # if self.show_animation:
        #     plt.pause(0.01)

        # animation_save_dir = "{}/results/tmp/".format(os.path.dirname(os.path.abspath(__file__)))
        # os.makedirs(animation_save_dir, exist_ok=True)
        # plt.savefig(animation_save_dir+"tmp_{}.png".format(str(iteration).zfill(6)))


class ClosedLoopNoPartitioner(ClosedLoopPartitioner):
    def __init__(self, At=None, bt=None, ct=None):
        ClosedLoopPartitioner.__init__(self, At=At, bt=bt, ct=ct)

class ClosedLoopUniformPartitioner(ClosedLoopPartitioner):
    def __init__(self, num_partitions=16, At=None, bt=None, ct=None):
        ClosedLoopPartitioner.__init__(self, At=At, bt=bt, ct=ct)
        self.num_partitions = num_partitions
        self.interior_condition = "linf"
        self.show_animation = False
        self.make_animation = False

    def get_one_step_reachable_set(self, A_inputs, b_inputs, A_out, propagator, num_partitions=None):
        reachable_set, info = self.get_reachable_set(A_inputs, b_inputs, A_out, propagator, t_max=1, num_partitions=num_partitions)
        return reachable_set, info

    def get_reachable_set(self, A_inputs, b_inputs, A_out, propagator, t_max, num_partitions=None):
        info = {}
        num_propagator_calls = 0

        # only used to compute slope in non-closedloop manner...
        input_polytope_verts = pypoman.duality.compute_polytope_vertices(A_inputs, b_inputs)
        input_range = np.empty((A_inputs.shape[1],2))
        input_range[:,0] = np.min(np.stack(input_polytope_verts), axis=0)
        input_range[:,1] = np.max(np.stack(input_polytope_verts), axis=0)

        input_shape = input_range.shape[:-1]
        if num_partitions is None:
            num_partitions = np.ones(input_shape, dtype=int)
            if isinstance(self.num_partitions, np.ndarray) and input_shape == self.num_partitions.shape:
                num_partitions = self.num_partitions
            elif len(input_shape) > 1:
                num_partitions[0,0] = self.num_partitions
            else:
                num_partitions *= self.num_partitions
        slope = np.divide((input_range[...,1] - input_range[...,0]), num_partitions)
        
        ranges = []
        reachable_set = None

        for element in product(*[range(num) for num in num_partitions.flatten()]):
            element_ = np.array(element).reshape(input_shape)
            input_range_ = np.empty_like(input_range)
            input_range_[...,0] = input_range[...,0]+np.multiply(element_, slope)
            input_range_[...,1] = input_range[...,0]+np.multiply(element_+1, slope)

            # This is a disaster hack to partition polytopes
            A_rect, b_rect = init_state_range_to_polytope(input_range_)
            rectangle_verts = pypoman.polygon.compute_polygon_hull(A_rect, b_rect)
            input_polytope_verts = pypoman.polygon.compute_polygon_hull(A_inputs, b_inputs)
            partition_verts = pypoman.intersection.intersect_polygons(input_polytope_verts, rectangle_verts)
            A_inputs_, b_inputs_ = pypoman.duality.compute_polytope_halfspaces(partition_verts)

            reachable_set_, info_ = propagator.get_reachable_set(A_inputs_, b_inputs_, A_out, t_max)
            num_propagator_calls += 1

            if reachable_set is None:
                reachable_set = np.stack(reachable_set_)

            # TODO: does this work?
            tmp = np.dstack([reachable_set, np.stack(reachable_set_)])
            reachable_set = np.max(tmp, axis=-1)
            
            ranges.append((input_range_, reachable_set_))

        info["all_partitions"] = ranges
        info["num_propagator_calls"] = num_propagator_calls
        info["num_partitions"] = np.product(num_partitions)

        return reachable_set, info
