import pickle
import numpy as np
import math

from scipy.spatial import ConvexHull, Delaunay

def save_dataset(xs, us):
    with open("dataset.pkl", "wb") as f:
        pickle.dump([xs, us], f)


def load_dataset():
    with open("/Users/mfe/Downloads/dataset.pkl", "rb") as f:
        xs, us = pickle.load(f)
    return xs, us


def range_to_polytope(state_range):
    num_states = state_range.shape[0]
    A = np.vstack([np.eye(num_states), -np.eye(num_states)])
    b = np.hstack([state_range[:, 1], -state_range[:, 0]])
    return A, b


def get_polytope_A(num):
    theta = np.linspace(0, 2 * np.pi, num=num + 1)
    A_out = np.dstack([np.cos(theta), np.sin(theta)])[0][:-1]
    return A_out


def get_next_state(xt, ut, At, bt, ct):
    return np.dot(At, xt.T) + np.dot(bt, ut.T)


def plot_polytope_facets(A, b, ls='-', show=True):
    import matplotlib.pyplot as plt
    cs = ['r','g','b','brown','tab:red', 'tab:green', 'tab:blue', 'tab:brown']
    ls = ['-', '-', '-', '-', '--', '--', '--', '--']
    num_facets = b.shape[0]
    x = np.linspace(1, 5, 2000)
    for i in range(num_facets):
        alpha = 0.2
        if A[i, 1] == 0:
            offset = -0.1*np.sign(A[i, 0])
            plt.axvline(x=b[i]/A[i, 0], ls=ls[i], c=cs[i])
            plt.fill_betweenx(y=np.linspace(-2, 2, 2000), x1=b[i]/A[i, 0], x2=offset+b[i]/A[i, 0], fc=cs[i], alpha=alpha)
        else:
            offset = -0.1*np.sign(A[i, 1])
            y = (b[i] - A[i, 0]*x)/A[i, 1]
            plt.plot(x, y, ls=ls[i], c=cs[i])
            plt.fill_between(x, y, y+offset, fc=cs[i], alpha=alpha)
    if show:
        plt.show()

def get_polytope_verts(A, b):
    import pypoman
    # vertices = pypoman.duality.compute_polytope_vertices(A, b)
    vertices = pypoman.polygon.compute_polygon_hull(A, b)
    print(vertices)

def is_in_convex_hull(p, hull):
    """
    Reference:
    https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
    
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0
def is_hull1_a_subset_of_hull2(hull_1, hull_2):
    pts_1 = hull_1.points[hull_1.vertices,:]
    hull_2_delaunay = Delaunay(hull_2.points)
    for pt in pts_1:
        if not(is_in_convex_hull(pt, hull_2_delaunay)):
            return False
    return True


def distance_point_to_segment(point, seg_pt_1, seg_pt_2):
    # https://stackoverflow.com/questions/41000123/computing-the-distance-to-a-convex-hull
    x1,y1 = seg_pt_1
    x2,y2 = seg_pt_2
    x3,y3 = point
    px = x2-x1
    py = y2-y1
    something = px*px + py*py + 1e-6
    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(something)
    if u > 1:
        u = 1
    elif u < 0:
        u = 0
    x = x1 + u * px
    y = y1 + u * py
    dx = x - x3
    dy = y - y3
    dist = np.sqrt(dx*dx + dy*dy)
    return dist
def distance_point_to_convex_hull(point, hull):
    pts_hull = hull.points[hull.vertices,:2]
    pts_hull = np.append(pts_hull, pts_hull[0,:][np.newaxis,:], axis=0)
    dists = np.zeros(pts_hull.shape[0])
    for i in range(pts_hull.shape[0]):
        if i<pts_hull.shape[0]-1:
            pt_hull_1 = pts_hull[i,:]
            pt_hull_2 = pts_hull[i+1,:]
        else:
            pt_hull_1 = pts_hull[-1,:]
            pt_hull_2 = pts_hull[0,:]
        dists[i] = distance_point_to_segment(point, pt_hull_1, pt_hull_2)
    return np.min(dists)
def Hausdorff_dist_two_convex_hulls(hull_1, hull_2):
    pts_hull_1 = hull_1.points[hull_1.vertices,:2]
    pts_hull_2 = hull_2.points[hull_2.vertices,:2]
    dists12 = np.zeros(pts_hull_1.shape[0])
    dists21 = np.zeros(pts_hull_2.shape[0])
    for i, pt in enumerate(pts_hull_1):
        dists12[i] = distance_point_to_convex_hull(pt, hull_2)
    for i, pt in enumerate(pts_hull_2):
        dists21[i] = distance_point_to_convex_hull(pt, hull_1)
    return np.maximum(np.max(dists12), np.max(dists21))
def are_points_in_ball(center, radius, points):
    # Inputs:
    #   center - (xdim,)
    #   radius - scalar
    #   points - (M,xdim)
    # Outputs:
    #   B_are_in_ball - (M,) (vector of booleans)
    return np.linalg.norm(points-center[None,:], axis=1)<=radius
def dist_points_to_ball(center, radius, points):
    # Inputs:
    #   center - (xdim,)
    #   radius - float
    #   points - (M,xdim)
    # Outputs:
    #   dists_to_ball - (M,) (vector of floats)
    return np.maximum(0, np.linalg.norm(points-center[None,:], axis=1)-radius)
def dist_points_to_sphere(center, radius, points):
    # Inputs:
    #   center - (xdim,)
    #   radius - float
    #   points - (M,xdim)
    # Outputs:
    #   dists_to_sphere - (M,) (vector of floats)
    return np.abs(np.linalg.norm(points-center[None,:], axis=1)-radius)
def Hausdorff_dist_ball_hull(ball_c, ball_r, hull):
    # Inputs:
    #   center - (xdim,)
    #   radius - float
    #   hull   - scipy.spatial.ConvexHull
    #   points - (M,xdim)
    # Outputs:
    #   haus_dist - float
    if ball_c.shape[0]>2:
        raise NotImplementedError("Not implemented for n_x > 2.")

    # dist_1 = dist_points_to_sphere(ball_c, ball_r, hull.points[hull.vertices,:])

    angs = np.arange(0, 2*np.pi, 0.01)
    sampled_pts = np.concatenate([ball_c[0,None,None]+ball_r*np.cos(angs)[:,None],
                                  ball_c[1,None,None]+ball_r*np.sin(angs)[:,None]], axis=1)

    # These lines won't work
    # dists_2 = np.zeros(sampled_pts.shape[0])
    # for i, pt in enumerate(sampled_pts):
    #     dists_2[i] = distance_point_to_convex_hull(pt, hull)

    # return np.maximum(np.max(dist_1), np.max(dists_2))
    hull_ball = ConvexHull(sampled_pts)

    return Hausdorff_dist_two_convex_hulls(hull, hull_ball)
def volume_n_ball(n=2, ball_r=1.0):
    np_pi, gamma = np.pi, math.gamma(n/2. + 1)
    vol = (np_pi**(n / 2.) / gamma) * (ball_r**n)
    return vol

# dists=[]
# for i in range(len(points)-1):
#     dists.append(dist(points[i][0],points[i][1],points[i+1][0],points[i+1][1],p[0],p[1]))
# dist = min(dists)

if __name__ == "__main__":
    test_1 = False
    test_2 = False
    test_3 = True

    import matplotlib.pyplot as plt

    if test_1:
        A = np.array([
                  [1, 1],
                  [0, 1],
                  [-1, -1],
                  [0, -1]
        ])
        b = np.array([2.8, 0.41, -2.7, -0.39])

        A2 = np.array([
                      [1, 1],
                      [0, 1],
                      [-0.97300157, -0.95230697],
                      [0.05399687, -0.90461393]
        ])
        b2 = np.array([2.74723146, 0.30446292, -2.64723146, -0.28446292])

        # get_polytope_verts(A, b)
        plot_polytope_facets(A, b)
        # get_polytope_verts(A2, b2)
        plot_polytope_facets(A2, b2, ls='--')
        plt.show()

    if test_2:
        pts_1 = np.array([[0.,0,3],
                          [0.,1,4],
                          [1.,1,5]])
        # pts_2 = np.array([[0.,0,3],
        #                   [0.,1,4],
        #                   [1.,1,5]])+1.
        pts_2 = np.array([[0.2,0.3,3],
                          [0.2,0.7,4],
                          [0.8,0.8,5]])
        hull_1 = ConvexHull(pts_1[:,:2])
        hull_2 = ConvexHull(pts_2[:,:2])

        pts = hull_1.points[hull_1.vertices,:2]
        pts = np.append(pts, pts[0,:][np.newaxis,:], axis=0)
        plt.plot(pts[:,0], 
                pts[:,1], 
                'g-', lw=1)
        pts = hull_2.points[hull_2.vertices,:2]
        pts = np.append(pts, pts[0,:][np.newaxis,:], axis=0)
        plt.plot(pts[:,0], 
                pts[:,1], 
                'b-', lw=1)
        plt.show()

        print("Hausdorff distance=", Hausdorff_dist_two_convex_hulls(hull_1, hull_2))
        print("hull1 is a subset of hull2=",is_hull1_a_subset_of_hull2(hull_1,hull_2))
        print("hull2 is a subset of hull1=",is_hull1_a_subset_of_hull2(hull_2,hull_1))

    if test_3:
        pts = np.array([[0.,0],
                        [0.,1],
                        [1.,1]])
        hull = ConvexHull(pts)

        ball_c = np.array([-0.,0.5])
        ball_r = 1.
        dH = Hausdorff_dist_ball_hull(ball_c, ball_r, hull)
        print("Hausdorff distance=", dH)


        pts = hull.points[hull.vertices,:2]
        pts = np.append(pts, pts[0,:][np.newaxis,:], axis=0)
        plt.plot(pts[:,0], 
                 pts[:,1], 
                 'b-', lw=1)
        ax = plt.gca()
        circle = plt.Circle((ball_c[0],ball_c[1]), ball_r, 
                            color='k', fill=False)
        ax.add_patch(circle)

        plt.show()