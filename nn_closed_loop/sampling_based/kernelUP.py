# We thank Adam Thorpe for discussing his kernel-based reachability
# analysis algorithm and helping debugging the implementation.
# - T. Lew, December 14, 2021

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

class kernelUP():
    def __init__(self, dynamics, 
                       controller, 
                       nb_samples=100, 
                       Lambda=0.1,
                       sigma=0.1
                ):
        self.dynamics    = dynamics
        self.controller  = controller
        self.nb_samples  = nb_samples
        self.Lambda      = Lambda
        self.sigma       = sigma
        # self.input_constraint = None # used for plotting

    def Abel_K(self, X1, X2=None):
        # Inputs:
        #   X1 - (M1,xdim)
        #   X2 - (M2,xdim) or None
        # Output:
        #   K  - (M1,M2) s.t. K[i,j] = exp(-||X1[i]-X2[j]||/sigma)
        # For reference, see 
        #   https://github.com/SheffieldML/GPy/blob/devel/GPy/kern/src/stationary.py
        if X2 is None:
            X2=X1
        sum_squares_X1X2 = np.sum(X1**2,1)[:,None]+np.sum(X2**2,1)[None,:]
        D_squared = sum_squares_X1X2-2.0*X1@X2.T
        D_squared = np.clip(D_squared, 0, np.inf)
        if X1 is X2:
            np.fill_diagonal(D_squared, 0)
        return np.exp(-np.sqrt(D_squared)/self.sigma)
    def Abel_classify(self, X1, X2):
        # Classifies X2 based on X1
        # Inputs:
        #   X1 - (M1,xdim)
        #   X2 - (M2,xdim) or None
        # Output:
        #   B_inside - (M2) vector of booleans, B_inside[i] True if 
        #               X2[i] is in within the decision boundary
        M1,xdim = X1.shape
        M2,dim2 = X2.shape
        if M1!=self.nb_samples:
            raise InputError("[Abel_classify] X1 should have self.nb_samples elements.")
        if xdim!=dim2:
            raise InputError("[Abel_classify] X1 and X2 should be of the same size.")

        G = self.Abel_K(X1)

        Winv = G + M1*self.Lambda*np.eye(M1)
        tau = 1 - np.min(np.diag(G.T@np.linalg.solve(Winv,G)))

        Phis         = self.Abel_K(X1, X2)
        W_Phis       = np.linalg.solve(Winv,Phis) # (M1,M2)
        PhisT_W_Phis = np.einsum('ij,ij->j', Phis, W_Phis)

        # Classify
        B_are_inside = (PhisT_W_Phis >= 1-tau)
        return B_are_inside

    def get_reachable_set(self, input_constraint, t_max, seed_id=0):
        # self.input_constraint = input_constraint
        # Sample reachable states
        dt = self.dynamics.dt
        num_timesteps = int((t_max + dt + np.finfo(float).eps)/dt)
        num_samples = num_timesteps*self.nb_samples
        xs, _ = self.dynamics.collect_data(
            t_max * self.dynamics.dt,
            input_constraint,
            num_samples=num_samples,
            controller =self.controller,
            merge_cols =False,
            seed_id=seed_id,
        )

        # New samples for reachable set computation
        xs_new, _ = self.dynamics.collect_data(
            t_max * self.dynamics.dt,
            input_constraint,
            num_samples=num_samples,
            controller =self.controller,
            merge_cols =False,
            seed_id=seed_id,
        )

        # Get reachable set from samples
        # reachable_set_fns = []
        reachable_set_xs = []
        for t in range(num_timesteps-1):
            # Classify new samples (for computation time estimate)
            self.Abel_classify(xs[:,t+1,:], xs_new[:,t+1,:])

            # ys of size (M,xdim)
            # reachable_set_classifier = lambda ys: self.Abel_classify(xs[:,t+1,:], ys)
            # reachable_set_fns.append(reachable_set_classifier)
            reachable_set_xs.append(xs[:,t+1,:])

        info = {}
        # return reachable_set_fns, info
        return reachable_set_xs, info

    def get_error(self, 
                  input_constraint, 
                  # reachable_set_fns, 
                  reachable_set_xs, 
                  t_max=5, 
                  seed_id=0):
        area_errors        = []
        haus_dists         = [] # hausdorff distance
        B_vec_conservative = [] # if true, the convex hull is conservative

        # Ground Truth
        # Samples to check conservatism
        xs_true_inside, _ = self.dynamics.collect_data(
            t_max * self.dynamics.dt,
            input_constraint,
            num_samples=10**6,
            controller =self.controller,
            merge_cols =False,
            seed_id=seed_id,
        )
        # Samples to check error w.r.t. convex hull
        true_hulls = self.dynamics.get_sampled_convex_hull(
            input_constraint, 
            t_max * self.dynamics.dt, 
            controller=self.controller
        )

        # error
        for t in range(t_max):
            true_hull = true_hulls[t+1]
            true_area = true_hull.volume

            # Compute approximate convex hull
            x_min = np.min(reachable_set_xs[t], 0)-0.2
            x_max = np.max(reachable_set_xs[t], 0)+0.2
            # mesh_xx = np.arange(x_min[0], x_max[0], 0.005)
            # mesh_xy = np.arange(x_min[1], x_max[1], 0.005)
            mesh_xx = np.linspace(x_min[0], x_max[0], 200)
            mesh_xy = np.linspace(x_min[1], x_max[1], 200)
            M_new_xx = mesh_xx.shape[0]
            M_new_xy = mesh_xy.shape[0]
            mesh_xx, mesh_xy = np.meshgrid(mesh_xx, mesh_xy)
            mesh_xs = np.concatenate([mesh_xx.reshape((M_new_xx*M_new_xy,1)),
                                      mesh_xy.reshape((M_new_xx*M_new_xy,1))], 
                                     axis=1)
            B_are_inside = self.Abel_classify(reachable_set_xs[t], mesh_xs)

            pts_inside = np.concatenate([mesh_xs[B_are_inside,0][:,None],
                                         mesh_xs[B_are_inside,1][:,None]], 
                                        axis=1)
            estimated_hull = ConvexHull(pts_inside)

            # check area coverage
            estimated_area = estimated_hull.volume
            area_errors.append((estimated_area - true_area) / true_area)

            # compute 2d Hausdorff distance
            true_hull = ConvexHull(true_hull.points[true_hull.vertices,:2])
            est_hull  = ConvexHull(estimated_hull.points[estimated_hull.vertices,:2])
            haus_dist = Hausdorff_dist_two_convex_hulls(true_hull, est_hull)
            haus_dists.append(haus_dist)

            # check if conservative outer approximation
            # here, use samples, not convex hull
            B_are_inside = self.Abel_classify(reachable_set_xs[t], 
                                              xs_true_inside[:,t+1,:])
            B_conservative = np.all(B_are_inside)
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


    def visualize(self, ax, input_dims, 
                  # reachable_set_fns, 
                  reachable_set_xs, 
                  iteration=0, title=None, 
                  reachable_set_color=None, 
                  reachable_set_zorder=None, 
                  reachable_set_ls=None, 
                  dont_tighten_layout=False,
                  B_show_label=False,
                  t_max=5):
        self.plot_reachable_sets(ax, 
                                 # reachable_set_fns, 
                                 reachable_set_xs, 
                                 input_dims, 
                                 reachable_set_color=reachable_set_color, 
                                 reachable_set_zorder=reachable_set_zorder, 
                                 reachable_set_ls=reachable_set_ls,
                                 B_show_label=B_show_label,
                                 t_max=t_max)

    def plot_reachable_sets(self, 
                            ax, 
                            # reachable_set_fns, 
                            reachable_set_xs, 
                            dims, 
                            reachable_set_color='tab:blue', 
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

        print("[kernelUP::plot_reachable_sets] plotting")

        # New samples for reachable set computation (ground truth)
        # if self.input_constraint is None:
        #     raise ValueError("Need to compute reachable sets first.")

        # get some samples to know where to evaluate classifier for plotting
        # xs_true_inside, _ = self.dynamics.collect_data(
        #     t_max * self.dynamics.dt,
        #     self.input_constraint,
        #     num_samples=10**4,
        #     controller =self.controller,
        #     merge_cols =False,
        #     seed_id=seed_id,
        # )

        # plot
        # for t in range(len(reachable_set_fns)):
        for t in range(len(reachable_set_xs)):
            if t < t_max:
                # classifier = reachable_set_fns[t]

                # where to create a grid for plotting
                x_min = np.min(reachable_set_xs[t], 0)-0.2
                x_max = np.max(reachable_set_xs[t], 0)+0.2

                # plot contour using meshgrid
                mesh_xx = np.arange(x_min[0], x_max[0], 0.005)
                mesh_xy = np.arange(x_min[1], x_max[1], 0.005)
                M_new_xx = mesh_xx.shape[0]
                M_new_xy = mesh_xy.shape[0]
                mesh_xx, mesh_xy = np.meshgrid(mesh_xx, mesh_xy)
                mesh_xs = np.concatenate([mesh_xx.reshape((M_new_xx*M_new_xy,1)),
                                          mesh_xy.reshape((M_new_xx*M_new_xy,1))], 
                                         axis=1)
                # vals = classifier(mesh_xs)
                vals = self.Abel_classify(reachable_set_xs[t], mesh_xs)

                mesh_xs_x = mesh_xs[:,0].reshape((M_new_xy,M_new_xx))
                mesh_xs_y = mesh_xs[:,1].reshape((M_new_xy,M_new_xx))
                vals      = vals.reshape((M_new_xy,M_new_xx)).astype(float)

                if B_show_label and t==0:
                    CS = ax.contour(mesh_xs_x,mesh_xs_y,vals,
                               levels=1, 
                               colors=reachable_set_color,
                               linewidths=2)#,
                               # linestyles='dashed')
                    # plt.clabel(CS, inline=1, fontsize=10)
                    CS.collections[0].set_label("Kernel")

                else:
                    ax.contour(mesh_xs_x,mesh_xs_y,vals,
                        levels=0, 
                        colors=reachable_set_color,
                        linewidths=2)#,
                        # linestyles='dashed')