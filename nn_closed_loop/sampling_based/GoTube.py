import numpy as np
import nn_closed_loop.constraints as constraints
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
from copy import deepcopy
from nn_closed_loop.utils.utils import are_points_in_ball
from nn_closed_loop.utils.utils import Hausdorff_dist_ball_hull
from nn_closed_loop.utils.utils import volume_n_ball

from scipy.spatial import ConvexHull

class GoTube():
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
        # print("[GoTube] computing reachable sets.")

        # get initial center
        center = (input_constraint.range[:,0]+input_constraint.range[:,1])/2.0

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
        _, num_timesteps, num_states = xs.shape

        # Get reachable set from samples
        reachable_sets = []
        for t in range(num_timesteps-1):
            # propagate center
            centers_now_next = np.concatenate([center[None,None,:],center[None,None,:]], axis=1)
            centers, _ = self.dynamics.run(1 * self.dynamics.dt,
                              input_constraint,
                              controller=self.controller,
                              merge_cols=False,
                              xs_presampled=centers_now_next)
            center = centers[0,1,:]

            # get radius
            r = np.max(np.linalg.norm(xs[:,t+1,:]-center[None,:], axis=1))
            r = r + self.padding_eps
            # Note: from the original paper https://arxiv.org/abs/2107.08467, one 
            # should compute, for some mu > 1:
            #     r = mu * np.max(np.linalg.norm(xs[:,t+1,:]-center[None,:], axis=1))
            # We use this slightly different radius computation to make it easier
            # to compare with epsilon-RandUP (changing it should not be an issue).

            # save result
            ball = [center,r]
            reachable_sets.append(ball)


        info = {}
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
            ball = reachable_sets[t]
            center, radius = ball

            # check area coverage
            true_area = true_hull.volume
            estimated_area = volume_n_ball(center.shape[0], radius)
            area_errors.append((estimated_area - true_area) / true_area)

            # compute 2d Hausdorff distance
            haus_dist = Hausdorff_dist_ball_hull(center, radius, true_hull)
            haus_dists.append(haus_dist)

            # check if conservative outer approximation
            points = true_hull.points[true_hull.vertices,:]
            B_are_inside = are_points_in_ball(center, radius, points)
            B_vec_conservative.append(np.all(B_are_inside))

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


    def visualize(self, ax, input_dims, 
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

        print("[GoTube::plot_reachable_sets] plotting")

        for t, ball in enumerate(reachable_sets):
            if t < t_max:
                center, radius = ball
                if B_show_label and t==0:
                    circle = plt.Circle((center[0],center[1]), radius, 
                                color=reachable_set_color, 
                                fill=False,
                                lw=2,
                                linestyle='dashed',
                                label="GoTube")
                else:
                    circle = plt.Circle((center[0],center[1]), radius, 
                                color=reachable_set_color, 
                                fill=False,
                                lw=2,
                                linestyle='dashed')
                ax.add_patch(circle)