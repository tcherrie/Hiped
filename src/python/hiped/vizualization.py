# vizualization.py
"""
A module containing a collection of useful functions for vizualization

Functions:
---------
getTransformedPlane(plane1, plane1]
    Used for plotting the faces of a 3D polytope
    
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def get_transformed_plan(plan1, plan2):
    """
    Transforms two 3D planes and returns a transformation matrix.
    
    Parameters:
    ----------
    plan1 : numpy.ndarray
        A 3x3 matrix representing the first plane.
    plan2 : numpy.ndarray
        A 3x3 matrix representing the second plane.
        
    Returns:
    -------
    numpy.ndarray
        A 4x4 transformation matrix.
    """
    x1, x2, x3 = plan1[:, 0]
    y1, y2, y3 = plan1[:, 1]
    z1, z2, z3 = plan1[:, 2]

    a1, a2, a3 = plan2[:, 0]
    b1, b2, b3 = plan2[:, 1]
    c1, c2, c3 = plan2[:, 2]

    t2 = x1 * y2 * z3
    t3 = x1 * y3 * z2
    t4 = x2 * y1 * z3
    t5 = x2 * y3 * z1
    t6 = x3 * y1 * z2
    t7 = x3 * y2 * z1
    t8 = -t3
    t9 = -t4
    t10 = -t7
    t11 = t2 + t5 + t6 + t8 + t9 + t10
    t12 = 1.0 / t11

    mt1 = np.array([
        t12 * (a1 * y2 * z3 - a1 * y3 * z2 - a2 * y1 * z3 + a2 * y3 * z1 + a3 * y1 * z2 - a3 * y2 * z1),
        t12 * (b1 * y2 * z3 - b1 * y3 * z2 - b2 * y1 * z3 + b2 * y3 * z1 + b3 * y1 * z2 - b3 * y2 * z1),
        t12 * (c1 * y2 * z3 - c1 * y3 * z2 - c2 * y1 * z3 + c2 * y3 * z1 + c3 * y1 * z2 - c3 * y2 * z1),
        t12 * (y1 * z2 - y2 * z1 - y1 * z3 + y3 * z1 + y2 * z3 - y3 * z2),
        -t12 * (a1 * x2 * z3 - a1 * x3 * z2 - a2 * x1 * z3 + a2 * x3 * z1 + a3 * x1 * z2 - a3 * x2 * z1),
        -t12 * (b1 * x2 * z3 - b1 * x3 * z2 - b2 * x1 * z3 + b2 * x3 * z1 + b3 * x1 * z2 - b3 * x2 * z1)
    ])

    mt2 = np.array([
        -t12 * (c1 * x2 * z3 - c1 * x3 * z2 - c2 * x1 * z3 + c2 * x3 * z1 + c3 * x1 * z2 - c3 * x2 * z1),
        -t12 * (x1 * z2 - x2 * z1 - x1 * z3 + x3 * z1 + x2 * z3 - x3 * z2),
        t12 * (a1 * x2 * y3 - a1 * x3 * y2 - a2 * x1 * y3 + a2 * x3 * y1 + a3 * x1 * y2 - a3 * x2 * y1),
        t12 * (b1 * x2 * y3 - b1 * x3 * y2 - b2 * x1 * y3 + b2 * x3 * y1 + b3 * x1 * y2 - b3 * x2 * y1),
        t12 * (c1 * x2 * y3 - c1 * x3 * y2 - c2 * x1 * y3 + c2 * x3 * y1 + c3 * x1 * y2 - c3 * x2 * y1),
        t12 * (x1 * y2 - x2 * y1 - x1 * y3 + x3 * y1 + x2 * y3 - x3 * y2),
        0.0,
        0.0,
        0.0,
        0.0
    ])
    return np.transpose(np.reshape(np.concatenate((mt1, mt2)), (4, 4)))



def legend2color(legend):
    """
    Assign colors to vertices of domains based on predefined rules.
    
    Parameters:
    ----------
    legend : list of str
        A list of labels for which colors need to be assigned.
        
    Returns:
    -------
    numpy.ndarray
        An array of RGB triplets corresponding to the input labels.
      
    Example usage:
    -------   
    legend = ["ap", "bm", "cp", "iron", "mat23", "truc"]
    colors = legend2color(legend)
    
    print(colors)
    """
    
    cycle_size = 10
    n_cycle = 4
    default_color = np.array([0.3, 0.3, 0.3])
    white_labels = ["air", "void"]
    black_labels = ["iron", "steel", "fe", "fesi", "feco", "feni", "smc"]
    color_prefix = ["mag", "v", "mat", "f", "n"]

    n_color = cycle_size * n_cycle
    cmap = plt.get_cmap('hsv', n_color)  # Other colormaps are possible: 'jet', etc.
    sample_list = np.arange(cycle_size*n_cycle).reshape(cycle_size, n_cycle).transpose().reshape(-1)
    cmap = cmap(sample_list)

    # RGB triplet of predefined labels
    defined_color = {
        "ap": np.array([1, 0, 0]), "am": np.array([1, 0, 1]),
        "bp": np.array([0, 1, 0]), "bm": np.array([1, 1, 0]),
        "cp": np.array([0, 0, 1]), "cm": np.array([0, 1, 1])
    }

    legend = [str(leg).lower() for leg in legend]
    color = np.ones((len(legend), 3)) * default_color

    for i, l in enumerate(legend):
        if l in white_labels:
            color[i, :] = np.array([1, 1, 1])
        elif l in black_labels:
            color[i, :] = np.array([0, 0, 0])
        elif l in defined_color:
            color[i, :] = defined_color[l]
        elif any(prefix in l for prefix in color_prefix):
            digits = ''.join(filter(str.isdigit, l))
            if digits:
                n = int(digits)
                color[i, :] = cmap[(n - 1) % n_color, :3]  # cmap returns RGBA, we need RGB
            else:
                print(f"Warning: '{l}' label color unknown, default color.")
        else:
            print(f"Warning: '{l}' label color unknown, default color.")

    color = color - 0.5  # for mixture with the other color
    return color





def plotcolor2D(p, elts, data, alpha=None):
    """
    Plot 2D color patches using vertex data and optional transparency.

    Parameters:
    ----------
    p : numpy.ndarray
        A 2D array where each row represents a point (x, y) in 2D space.
    elts : numpy.ndarray
        A 2D array where each row represents an element defined by indices into `p`.
    data : list or numpy.ndarray
        Color data for each element.
    alpha : numpy.ndarray or None
        Optional transparency values for each element.
    
    Example usage:
    -------   
    p = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    elts = np.array([[0, 1, 2], [1, 2, 3]])
    data = ['red', 'blue']
    alpha = [0.5, 0.7]

    plotcolor2D(p, elts, data, alpha)
    """
    elt1, elt2, elt3 = elts[:, 0], elts[:, 1], elts[:, 2]
    x = np.array([p[elt1, 0], p[elt2, 0], p[elt3, 0]])
    y = np.array([p[elt1, 1], p[elt2, 1], p[elt3, 1]])

    fig, ax = plt.subplots()
    patches = []
    for i in range(x.shape[1]):
        polygon = Polygon(np.array([x[:, i], y[:, i]]).T, closed=True)
        patches.append(polygon)

    collection = PatchCollection(patches, facecolor=data, edgecolor='none')

    if alpha is not None:
        collection.set_alpha(alpha)

    ax.add_collection(collection)
    ax.autoscale_view()
    plt.show()
    


def rot3D(ax, th):
    """
    Generates a 3D rotation matrix around a given axis by a specified angle.

    Parameters:
    ----------
    ax : numpy.ndarray
        A 3-element array representing the axis of rotation.
    th : float
        The angle of rotation in radians.

    Returns:
    -------
    numpy.ndarray
        A 3x3 rotation matrix.
        
    Example usage:
    -------
    axis = np.array([1, 1, 0])
    angle = np.pi / 4  # 45 degrees in radians

    rotation_matrix = rot3D(axis, angle)
    print(rotation_matrix)
    """
    # Normalize the axis vector
    u = ax / np.linalg.norm(ax)
    ux, uy, uz = u

    # Calculate cosine and sine of the angle
    c = np.cos(th)
    s = np.sin(th)

    # Compute the rotation matrix
    R = np.array([
        [ux**2 * (1 - c) + c, ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
        [ux * uy * (1 - c) + uz * s, uy**2 * (1 - c) + c, uy * uz * (1 - c) - ux * s],
        [ux * uz * (1 - c) - uy * s, uy * uz * (1 - c) + ux * s, uz**2 * (1 - c) + c]
    ])
    
    return R

def rotPoint(p, th1, th2, th3):
    """
    Applies a sequence of 3D rotations to a point.

    Parameters:
    ----------
    p : numpy.ndarray
        A 3-element array representing the point to rotate.
    th1 : float
        The angle of rotation around the x-axis in radians.
    th2 : float
        The angle of rotation around the y-axis in radians.
    th3 : float
        The angle of rotation around the z-axis in radians.

    Returns:
    -------
    numpy.ndarray
        The rotated point.
    
    Example usage:
    ------
    point = np.array([1, 0, 0])
    angle1 = np.pi / 6  # 30 degrees in radians
    angle2 = np.pi / 4  # 45 degrees in radians
    angle3 = np.pi / 3  # 60 degrees in radians

    rotated_point = rotPoint(point, angle1, angle2, angle3)
    print(rotated_point)
    """
    # Apply the rotations
    R1 = rot3D(np.array([1, 0, 0]), th1)
    R2 = rot3D(np.array([0, 1, 0]), th2)
    R3 = rot3D(np.array([0, 0, 1]), th3)
    
    # Combine the rotations
    R = R3 @ R2 @ R1
    
    # Apply the combined rotation matrix to the point
    p_rotated = R @ p
    
    return p_rotated



def pointTriangle(pp,l=0, p=[]):
    if l==0 :
        p += pp
        return p
    else :
        l -= 1
        p1, p2, p3 = pp[0], pp[1], pp[2]
        p = pointTriangle([(p1+p2)/2,(p1+p3)/2,p1],l,p)
        p = pointTriangle([(p1+p2)/2,(p2+p3)/2,p2],l,p)
        p = pointTriangle([(p1+p3)/2,(p2+p3)/2,p3],l,p)
        p = pointTriangle([(p1+p2)/2,(p2+p3)/2,(p1+p3)/2],l,p)
    return p

