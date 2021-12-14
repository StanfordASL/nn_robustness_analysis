import numpy as np
import nn_closed_loop.constraints as constraints
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
from copy import deepcopy
import os

from nn_closed_loop.utils.utils import Hausdorff_dist_two_convex_hulls
from nn_closed_loop.utils.utils import is_hull1_a_subset_of_hull2
from scipy.spatial import ConvexHull

class randUP():
    def __init__(self, dynamics, 
                       controller, 
                       nb_samples=1e5, 
                       padding_eps=0.01
                ):
        self.dynamics    = dynamics
        self.controller  = controller
        self.nb_samples  = nb_samples
        self.padding_eps = padding_eps

    def get_reachable_set(self, input_constraint, t_max, seed_id=0):
        # Sample reachable states
        dt = self.dynamics.dt
        num_timesteps = int((t_max + dt + np.finfo(float).eps)/dt)
        num_samples = num_timesteps*self.nb_samples
        xs, us = self.dynamics.collect_data(
            t_max * self.dynamics.dt,
            input_constraint,
            num_samples=num_samples,
            controller =self.controller,
            merge_cols =False,
            seed_id=seed_id,
        )
        num_runs, num_timesteps, num_states = xs.shape

        # Get reachable set from samples
        reachable_sets = []
        for t in range(num_timesteps-1):
            # Convex hull
            hull = ConvexHull(xs[:,t+1,:]) # last dim is state dim.

            if self.padding_eps>0:
                if xs.shape[-1] == 2:
                    # Points on circle
                    r        = self.padding_eps
                    thetas   = np.arange(0, 2*np.pi, 0.1)
                    pts_ball = np.concatenate((r*np.cos(thetas)[:,np.newaxis], 
                                               r*np.sin(thetas)[:,np.newaxis]), 
                                              axis=1)
                    # Note that directly epsilon-padding the reachable set estimates
                    # is not strictly necessary --- e.g., for trajectory optimization,
                    # one could simply pad all constraints by epsilon and use the 
                    # unpadded (epsilon=0) reachable set estimate from RandUP. 
                else:
                    raise NotImplementedError("eps-padding not implemented for n_x > 2.")

                # Epsilon-padded convex hull
                pts  = xs[hull.vertices,t+1,:2]
                N, M = pts.shape[0], len(thetas)
                pts  = np.tile(pts.T, M).T + np.tile(pts_ball.T, N).T # Minkowski sum
                hull = ConvexHull(pts)

            # save result
            reachable_sets.append(hull)



        info = {}
        # return output_constraint, info
        return reachable_sets, info

    def get_error(self, input_constraint, reachable_sets, t_max=5):
        area_errors        = []
        haus_dists         = [] # hausdorff distance
        B_vec_conservative = [] # if true, the convex hull is conservative

        # ground truth
        true_hulls = self.dynamics.get_sampled_convex_hull(input_constraint, 
                                                           t_max * self.dynamics.dt, 
                                                           controller=self.controller)

        # error
        for t in range(t_max):
            true_hull = true_hulls[t+1]
            estimated_hull = reachable_sets[t]

            # check area coverage
            true_area = true_hull.volume
            estimated_area = estimated_hull.volume
            area_errors.append((estimated_area - true_area) / true_area)


            # compute 2d Hausdorff distance
            true_hull = ConvexHull(true_hull.points[true_hull.vertices,:2])
            est_hull  = ConvexHull(estimated_hull.points[estimated_hull.vertices,:2])
            haus_dist = Hausdorff_dist_two_convex_hulls(true_hull, est_hull)
            haus_dists.append(haus_dist)
            # print("Hausdorff_dist=",haus_dist)

            # check if conservative outer approximation
            B_conservative = is_hull1_a_subset_of_hull2(true_hull, est_hull)
            B_vec_conservative.append(B_conservative)

        final_area_error = area_errors[-1]
        avg_area_error = np.mean(area_errors)
        final_haus_error = haus_dists[-1]
        avg_haus_error = np.mean(haus_dists)
        B_all_conservative = np.all(B_vec_conservative)

        errors = ([final_area_error, avg_area_error, np.array(area_errors),
                  final_haus_error, avg_haus_error, np.array(haus_dists)],
                  [B_vec_conservative, B_all_conservative]
                  )
        return errors


    def visualize(self, ax, 
                        input_dims, 
                        reachable_sets, 
                        iteration=0, 
                        title=None, 
                        reachable_set_color='k', 
                        reachable_set_zorder=None, 
                        reachable_set_ls=None, 
                        dont_tighten_layout=False,
                        B_show_label=False,
                        t_max=5):
        self.plot_reachable_sets(ax, 
                                 reachable_sets, 
                                 input_dims, 
                                 reachable_set_color=reachable_set_color, 
                                 reachable_set_zorder=reachable_set_zorder, 
                                 reachable_set_ls=reachable_set_ls, 
                                 B_show_label=B_show_label,
                                 t_max=t_max)

    def plot_reachable_sets(self, 
                            ax, 
                            reachable_sets, 
                            dims, 
                            reachable_set_color="tab:blue", 
                            reachable_set_zorder=None, 
                            reachable_set_ls="-",
                            B_show_label=False,
                            t_max=5):
        fc_color = "None"

        if len(dims) == 2:
            projection = None
            self.plot_2d = True
            self.linewidth = 3
        elif len(dims) == 3:
            projection = '3d'
            self.plot_2d = False
            self.linewidth = 1

        print("[randup::plot_reachable_sets] plotting")

        for t, hull in enumerate(reachable_sets):
            if t < t_max:
                pts = hull.points[hull.vertices,:2]
                pts = np.append(pts, pts[0,:][np.newaxis,:], axis=0)

                if B_show_label and t==0:
                    ax.plot(pts[:,0], 
                            pts[:,1], 
                            lw=2,
                            color="tab:orange",#reachable_set_color, 
                            label="RandUP",
                            linestyle='dashed')
                else:
                    ax.plot(pts[:,0], 
                            pts[:,1], 
                            lw=2,
                            color="tab:orange",#reachable_set_color, 
                            linestyle='dashed')