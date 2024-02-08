import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from nn_closed_loop.utils.nn_bounds import BoundClosedLoopController
from nn_closed_loop.utils.utils import sample_pts_unit_ball, sample_pts_in_ellipsoid
import nn_closed_loop
import nn_closed_loop.constraints as constraints
import torch
import os
import pickle

from scipy.spatial import ConvexHull

dir_path = os.path.dirname(os.path.realpath(__file__))


class Dynamics:
    def __init__(
        self,
        At,
        bt,
        ct,
        u_limits=None,
        dt=1.0,
        c=None,
        sensor_noise=None,
        process_noise=None,
    ):

        # State dynamics
        self.At = At
        self.bt = bt
        self.ct = ct
        self.num_states, self.num_inputs = bt.shape

        # Observation Dynamics and Noise
        if c is None:
            c = np.eye(self.num_states)
        self.c = c
        self.num_outputs = self.c.shape[0]
        self.sensor_noise = sensor_noise
        self.process_noise = process_noise

        # Min/max control inputs
        self.u_limits = u_limits

        self.dt = dt

        # functions for generating extremal trajectories
        if not(isinstance(self, nn_closed_loop.dynamics.DoubleIntegrator)):
            raise NotImplementedError
        if self.num_states != 2:
            raise NotImplementedError
        if self.num_inputs != 1:
            raise NotImplementedError
        noise_magnitude = process_noise[0, 1] # infinity norm
        # ||w||_2 <= sqrt(state_dim) * ||w||_\infty
        # Note: self.w_ball_radius already accounts for dt,
        # see DoubleIntegrator.py
        self.w_ball_radius = np.sqrt(self.num_states) * noise_magnitude

    def control_nn(self, x, model):
        if x.ndim == 1:
            batch_x = np.expand_dims(x, axis=0)
        else:
            batch_x = x
        us = model.forward(torch.Tensor(batch_x)).data.numpy()
        return us

    def observe_step(self, xs):
        obs = np.dot(xs, self.c.T)
        if self.sensor_noise is not None:
            noise = np.random.uniform(
                low=self.sensor_noise[:, 0],
                high=self.sensor_noise[:, 1],
                size=xs.shape,
            )
            obs += noise
        return obs

    def dynamics_step(self, xs, us):
        raise NotImplementedError

    def colors(self, t_max):
        return [cm.get_cmap(self.cmap_name)(i) for i in range(t_max + 1)]

    def get_sampled_output_range(
        self, input_constraint, t_max=5, num_samples=1000, controller="mpc",
        output_constraint=None
    ):
        xs, us = self.collect_data(
            t_max,
            input_constraint,
            num_samples,
            controller=controller,
            merge_cols=False,
        )
        num_runs, num_timesteps, num_states = xs.shape
        if isinstance(input_constraint, constraints.PolytopeConstraint):
            # hack: just return all the sampled pts for error calculator
            sampled_range = xs
            # num_facets = output_constraint.A.shape[0]
            # all_pts = np.dot(output_constraint.A, xs.T.reshape(num_states, -1))
            # all_pts = all_pts.reshape(num_facets, num_runs, num_timesteps)
            # all_pts = all_pts[..., 1:]  # drop zeroth timestep
            # sampled_range = np.max(all_pts, axis=1).T
        elif isinstance(input_constraint, constraints.LpConstraint):
            sampled_range = np.zeros((num_timesteps - 1, num_states, 2))
            for t in range(1, num_timesteps):
                sampled_range[t - 1, :, 0] = np.min(xs[:, t, :], axis=0)
                sampled_range[t - 1, :, 1] = np.max(xs[:, t, :], axis=0)
        else:
            raise NotImplementedError
        return sampled_range

    def get_sampled_convex_hull(self,
        input_constraint, 
        t_max=5, 
        num_samples=2*10**6, 
        controller="mpc",
        method="monte_carlo"):
        if method=="monte_carlo":
            xs, us = self.collect_data(
                t_max,
                input_constraint,
                num_samples=num_samples,
                controller=controller,
                merge_cols=False,
            )
        elif method=="extremal":
            num_samples = 10**4
            xs, us = self.collect_extremal_data(
                t_max,
                input_constraint,
                num_samples=num_samples,
                controller=controller,
                merge_cols=False,
            )
        else:
            raise NotImplementedError
        num_runs, num_timesteps, num_states = xs.shape

        hulls = []
        for t in range(num_timesteps):
            hull = ConvexHull(xs[:,t,:2])
            hulls.append(hull)
        return hulls

    def show_samples(
        self,
        t_max,
        input_constraint,
        save_plot=False,
        ax=None,
        show=False,
        controller="mpc",
        input_dims=[[0], [1]],
        zorder=1,
        show_samples_labels=False,
    ):
        if ax is None:
            if len(input_dims) == 2:
                projection = None
            elif len(input_dims) == 3:
                projection = '3d'
            ax = plt.subplot(projection=projection)

        xs, us = self.collect_data(
            t_max,
            input_constraint,
            num_samples=10**6,
            controller=controller,
            merge_cols=False,
        )

        num_runs, num_timesteps, num_states = xs.shape
        colors = self.colors(num_timesteps)

        for t in range(num_timesteps):
            ax.scatter(
                *[xs[:, t, i] for i in input_dims],
                color=(0.3,0.3,0.3),
                s=0.5,
                alpha=0.05,
                zorder=zorder,
            )

        ax.set_xlabel(r"$x_" + str(input_dims[0][0]) + "$", fontsize=36)
        ax.set_ylabel(r"$x_" + str(input_dims[1][0]) + "$", fontsize=36)

        # plt.grid(which='minor', alpha=0.5, linestyle='--')
        # plt.grid(which='major', alpha=0.75, linestyle=':')
        ax.tick_params(axis='both', which='major', labelsize=26)
        ax.tick_params(axis='both', which='minor', labelsize=16)

        if len(input_dims) == 3:
            ax.set_zlabel(r"$x_" + str(input_dims[2][0]) + "$")

        if save_plot:
            ax.savefig(plot_name)

        if show:
            plt.show()

    def collect_data(
        self,
        t_max,
        input_constraint,
        num_samples=2420,
        controller="mpc",
        merge_cols=True,
        seed_id=0,
    ):
        xs, us = self.run(
            t_max,
            input_constraint,
            num_samples,
            collect_data=True,
            controller=controller,
            merge_cols=merge_cols,
            seed_id=seed_id,
        )
        return xs, us

    def run(
        self,
        t_max,
        input_constraint,
        num_samples=100,
        collect_data=False,
        clip_control=True,
        controller="mpc",
        merge_cols=False,
        seed_id=0,
        xs_presampled=None,
    ):
        
        np.random.seed(seed_id)
        num_timesteps = int(
            (t_max + self.dt + np.finfo(float).eps) / (self.dt)
        )
        
        if xs_presampled is None:
            if collect_data:
                num_runs = int(num_samples / num_timesteps)
                xs = np.zeros((num_runs, num_timesteps, self.num_states))
                us = np.zeros((num_runs, num_timesteps, self.num_inputs))
            
            # Initial state
            if isinstance(input_constraint, constraints.LpConstraint):
                if input_constraint.p == np.inf:
                    # xs[:, 0, :] = np.random.uniform(
                    #     low=input_constraint.range[:, 0],
                    #     high=input_constraint.range[:, 1],
                    #     size=(num_runs, self.num_states),
                    # )
                    ellipsoid_center = 0.5 * (
                        input_constraint.range[:, 0] + input_constraint.range[:, 1])
                    ellipsoid_delta = input_constraint.range[:, 1] - input_constraint.range[:, 0]
                    ellipsoid_delta = 0.5 * ellipsoid_delta # since distance to center
                    ellipsoid_Q = self.num_states * np.diag(ellipsoid_delta**2)
                    xs[:, 0, :] = sample_pts_in_ellipsoid(
                        ellipsoid_center, ellipsoid_Q, num_runs).T
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        else:
            xs = xs_presampled
            us = np.zeros((xs_presampled.shape[0], num_timesteps, self.num_inputs))

        t = 0
        step = 0
        while t < t_max:

            # Observe system (using observer matrix,
            # possibly adding measurement noise)
            obs = self.observe_step(xs[:, step, :])

            # Compute Control
            if controller == "mpc":
                u = self.control_mpc(x0=obs)
            elif isinstance(
                controller, BoundClosedLoopController
            ) or isinstance(controller, torch.nn.Sequential):
                # print("controller =", controller)
                u = self.control_nn(x=obs, model=controller)
            else:
                raise NotImplementedError
            if clip_control and (self.u_limits is not None):
                u = np.clip(u, self.u_limits[:, 0], self.u_limits[:, 1])

            # Step through dynamics (possibly adding process noise)
            xs[:, step + 1, :] = self.dynamics_step(xs[:, step, :], u)

            us[:, step, :] = u
            step += 1
            t += self.dt + np.finfo(float).eps

        if merge_cols:
            return xs.reshape(-1, self.num_states), us.reshape(
                -1, self.num_inputs
            )
        else:
            return xs, us

    def project(self, v, u):
        # projection onto the tangent space of the sphere
        # of radius $||v||$.
        u_projected = u - np.dot(v, u) * v
        return u_projected

    def project_batched(self, vs, us):
        us_projected = us - (np.einsum('Mi,Mi->M', vs, us) * vs.T).T
        return us_projected

    def n_W(self, w):
        # outward-pointing normal vector of 
        # $\partial\mathcal{W}$ at $w$
        normal_vector = w / np.linalg.norm(w)
        return normal_vector

    def n_W_batched(self, ws):
        # ws - (M, num_states)
        # outward-pointing normal vector of 
        # $\partial\mathcal{W}$ at $w$
        normal_vectors = (ws.T / np.linalg.norm(ws, axis=1)).T
        return normal_vectors

    def n_W_inverse(self, n):
        # w\in\partial\mathcal{W}$ such that n_w(w)=n
        w = self.w_ball_radius * n
        return w

    def n_W_inverse_batched(self, ns):
        # w\in\partial\mathcal{W}$ such that n_w(w)=n
        ws = self.w_ball_radius * ns
        return ws

    def n_X0(self, x0, c, Q, Qinv):
        normal_vector = Qinv @ (x0 - c)
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        return normal_vector

    def n_X0_batched(self, x0s, c, Q, Qinv):
        # x0s - (M, num_states)
        normal_vectors = Qinv @ (x0s - c).T
        normal_vectors = normal_vectors / np.linalg.norm(normal_vector, axis=0)
        normal_vectors = normal_vectors.T
        return normal_vectors

    def n_X0_inverse(self, d, c, Q, Qinv):
        x0 = c + Q @ d / np.sqrt(d.T @ Q @ d)
        return w

    def n_X0_inverse_batched(self, ds, c, Q, Qinv):
        # ds - (M, num_states)
        x0s = Q @ ds.T / np.sqrt(np.einsum('Mx,xy,My->M', ds, Q, ds))
        x0s = x0s.T + c
        return x0s

    def pmp_disturbances_ws_from_qs(self, qs):
        ds = -qs.T / np.linalg.norm(qs, axis=1)
        ws = self.n_W_inverse_batched(ds).T
        return ws

    def collect_extremal_data(
        self,
        t_max,
        input_constraint,
        num_samples=2420,
        controller="mpc",
        merge_cols=True,
        seed_id=0,
    ):
        np.random.seed(seed_id)
        if not(isinstance(self, nn_closed_loop.dynamics.DoubleIntegrator)):
            raise NotImplementedError
        if self.num_states != 2:
            raise NotImplementedError
        if self.num_inputs != 1:
            raise NotImplementedError

        dt = self.dt
        num_timesteps = int(
            (t_max + dt + np.finfo(float).eps) / dt
        )

        # dynamics are those of the double integrator
        dyns_dt = 0.25
        A_ct = np.array([[0, 1.], [0, 0]])
        b_ct = np.array([[0.5 * dyns_dt], [1]])
        A_dt = np.eye(2) + dyns_dt * A_ct
        b_dt = dyns_dt * b_ct
        
        num_samples = int(num_samples / num_timesteps)
        xs = np.zeros((num_samples, num_timesteps, self.num_states)) # states
        qs = np.zeros((num_samples, num_timesteps, self.num_states)) # augmented states
        us = np.zeros((num_samples, num_timesteps, self.num_inputs)) # controls

        # Initial state
        if isinstance(input_constraint, constraints.LpConstraint):
            if input_constraint.p == np.inf:
                xs[:, 0, :] = np.random.uniform(
                    low=input_constraint.range[:, 0],
                    high=input_constraint.range[:, 1],
                    size=(num_samples, self.num_states),
                )
                X0_ellipsoid_center = 0.5 * (
                    input_constraint.range[:, 0] + input_constraint.range[:, 1])
                X0_ellipsoid_delta = input_constraint.range[:, 1] - input_constraint.range[:, 0]
                X0_ellipsoid_delta = 0.5 * X0_ellipsoid_delta # since distance to center
                X0_ellipsoid_Q = self.num_states * np.diag(X0_ellipsoid_delta**2)
                X0_ellipsoid_Qinv = np.diag(1.0 / (self.num_states * X0_ellipsoid_delta**2))
            else:
                raise NotImplementedError

        # Initial directions
        thetas = np.linspace(0, 2*np.pi, num_samples)
        d0s = np.stack((np.cos(thetas), np.sin(thetas))).T # (num_samples, 2)
        # Initial states
        xs[:, 0, :] = self.n_X0_inverse_batched(-d0s, 
            X0_ellipsoid_center, X0_ellipsoid_Q, X0_ellipsoid_Qinv)
        # Initial augmented states
        qs[:, 0, :] = d0s

        t = 0
        step = 0
        while t < t_max:
            xs_t, qs_t = xs[:, step, :], qs[:, step, :]
            xs_t_tensor = torch.as_tensor(xs_t, dtype=torch.float32)
            xs_t_tensor.requires_grad = True
            ws_t = self.pmp_disturbances_ws_from_qs(qs_t)
            # Note: pmp_disturbances_ws_from_qs already accounts for dt

            # get control
            if isinstance(
                controller, BoundClosedLoopController
            ) or isinstance(controller, torch.nn.Sequential):
                us_t = controller.forward(xs_t_tensor)
                # this works since each ith entry of us_t (where i=1,...,num_samples)
                # only depends on the ith entry of xs_t_tensor 
                # and also since the control inputs are scalar, otherwise we would
                # need to loop over the number of control inputs
                us_t_dx = torch.cat(
                    torch.autograd.grad(us_t.sum(), xs_t_tensor, 
                    retain_graph=True))
                us_t_dx = us_t_dx[:, None, :] # scalar control input
                us_t = torch.clip(us_t, 
                    min=self.u_limits[0, 0], 
                    max=self.u_limits[0, 1])
            else:
                raise NotImplementedError
            us_t = us_t.data.numpy() # (num_samples, 1)
            us_t_dx = us_t_dx.data.numpy() # (num_samples, 1, 2)
    
            xs_next = (np.dot(A_dt, xs_t.T) + np.dot(b_dt, us_t.T)).T
            xs_next = xs_next + ws_t # note: ws_t already have dt included

            # chain rule to get Jacobian of f_dynamics(x, neural_net(x))
            fs_dx_times_qs = (np.einsum('xy,Mx->My', A_ct, qs_t) + 
               np.einsum('Mxy,Mx->My', np.einsum('xu,Muy->Mxy', b_ct, us_t_dx), qs_t))
            qs_dot = -fs_dx_times_qs
            qs_next = qs_t + dyns_dt * qs_dot

            # save data
            xs[:, step + 1, :] = xs_next
            qs[:, step + 1, :] = qs_next
            us[:, step, :] = us_t
            step += 1
            t += dt + np.finfo(float).eps

        if merge_cols:
            return xs.reshape(-1, self.num_states), us.reshape(
                -1, self.num_inputs
            )
        else:
            return xs, us

class ContinuousTimeDynamics(Dynamics):
    def __init__(
        self,
        At,
        bt,
        ct,
        u_limits=None,
        dt=1.0,
        c=None,
        sensor_noise=None,
        process_noise=None,
    ):
        super().__init__(At, bt, ct, u_limits, dt, c, sensor_noise, process_noise)
        self.continuous_time = True

    def dynamics(self, xs, us):
        xdot = (np.dot(self.At, xs.T) + np.dot(self.bt, us.T)).T + self.ct
        if self.process_noise is not None:
            noise = np.random.uniform(
                low=self.process_noise[:, 0],
                high=self.process_noise[:, 1],
                size=xs.shape,
            )
            xdot += noise
        return xdot

    def dynamics_step(self, xs, us):
        return xs + self.dt * self.dynamics(xs, us)


class DiscreteTimeDynamics(Dynamics):
    def __init__(
        self,
        At,
        bt,
        ct,
        u_limits=None,
        dt=1.0,
        c=None,
        sensor_noise=None,
        process_noise=None,
    ):
        super().__init__(At, bt, ct, u_limits, dt, c, sensor_noise, process_noise)
        self.continuous_time = False

    def dynamics_step(self, xs, us):
        xs_t1 = (np.dot(self.At, xs.T) + np.dot(self.bt, us.T)).T + self.ct
        if self.process_noise is not None:
            noise = sample_pts_unit_ball(xs.shape[1], xs.shape[0]).T # (M, x_dim)
            noise *= np.sqrt(xs.shape[1]) * self.process_noise[0, 1]
            xs_t1 += noise
        return xs_t1


if __name__ == "__main__":

    from nn_closed_loop.dynamics.DoubleIntegrator import DoubleIntegrator
    dynamics = DoubleIntegrator()
    init_state_range = np.array([
        # (num_inputs, 2)
        [2.5, 3.0],  # x0min, x0max
        [-0.1, 0.1], # x1min, x1max
    ])
    xs, us = dynamics.collect_data(
        t_max=10,
        input_constraint=constraints.LpConstraint(
            p=np.inf, range=init_state_range
        ),
        num_samples=2420,
    )
    print(xs.shape, us.shape)
    system = "double_integrator"
    with open(dir_path + "/../../datasets/{}/xs.pkl".format(system), "wb") as f:
        pickle.dump(xs, f)
    with open(dir_path + "/../../datasets/{}/us.pkl".format(system), "wb") as f:
        pickle.dump(us, f)

    # from nn_closed_loop.utils.nn import load_model

    # # dynamics = DoubleIntegrator()
    # # init_state_range = np.array([ # (num_inputs, 2)
    # #                   [2.5, 3.0], # x0min, x0max
    # #                   [-0.25, 0.25], # x1min, x1max
    # # ])
    # # controller = load_model(name='double_integrator_mpc')

    # dynamics = QuadrotorOutputFeedback()
    # init_state_range = np.array([ # (num_inputs, 2)
    #               [4.65,4.65,2.95,0.94,-0.01,-0.01], # x0min, x0max
    #               [4.75,4.75,3.05,0.96,0.01,0.01] # x1min, x1max
    # ]).T
    # goal_state_range = np.array([
    #                       [3.7,2.5,1.2],
    #                       [4.1,3.5,2.6]
    # ]).T
    # controller = load_model(name='quadrotor')
    # t_max = 15*dynamics.dt
    # input_constraint = LpConstraint(range=init_state_range, p=np.inf)
    # dynamics.show_samples(t_max, input_constraint, save_plot=False, ax=None, show=True, controller=controller)
