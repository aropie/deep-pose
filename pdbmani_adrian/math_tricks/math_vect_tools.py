import numpy as np
from math import acos, atan2, cos, sin,atan
from numpy import array, float64, zeros
from numpy.linalg import norm

def cartesian_to_spherical(vector):
    """Convert the Cartesian vector [x, y, z] to spherical coordinates [r, theta, phi].

    The parameter r is the radial distance, theta is the polar angle, and phi is the azimuth.


    @param vector:  The Cartesian vector [x, y, z].
    @type vector:   numpy rank-1, 3D array
    @return:        The spherical coordinate vector [r, theta, phi].
    @rtype:         numpy rank-1, 3D array
    """

    # The radial distance.
    r = norm(vector)

    # Unit vector.
    unit = vector / r

    # The polar angle.
    theta = acos(unit[2])

    # The azimuth.
    phi = atan2(unit[1], unit[0])

    # Return the spherical coordinate vector.
    return array([r, theta, phi], float64)

def create_vector2d(vec):
    """Returns a vector as a numpy array."""
    return np.array([vec[0],vec[1]])

def create_vector(vec):
    """Returns a vector as a numpy array."""
    return np.array([vec[0],vec[1],vec[2]])

def create_vectors(vec1,vec2,vec3,vec4):
    """Returns dihedral angle, takes four
    Scientific.Geometry.Vector objects
    (dihedral does not work for them because
    the Win and Linux libraries are not identical.
    """
    return map(create_vector,[vec1,vec2,vec3,vec4])

def normalize_vec(vec1):
    return vec1/np.linalg.norm(vec1)

def scalar(v1,v2):
    """
    calculates the scalar product of two vectors
    v1 and v2 are numpy.array objects.
    returns a float for a one-dimensional array.
    """
    return sum(v1*v2)

def angle(v1,v2):
    """
    calculates the angle between two vectors.
    v1 and v2 are numpy.array objects.
    returns a float containing the angle in radians.
    """
    length_product = np.linalg.norm(v1)*np.linalg.norm(v2)
    if length_product == 0:
        raise AngleGeometryError(\
        "Cannot calculate angle for vectors with length zero")
    cosine = scalar(v1,v2)/length_product
    angle = np.arccos(cosine)
    return angle

def calc_angle(vec1,vec2,vec3):
    """Calculates a flat angle from three coordinates."""
    if len(vec1) == 3:
        v1, v2, v3 = map(create_vector,[vec1,vec2,vec3])
    else:
        v1, v2, v3 = map(create_vector2d,[vec1,vec2,vec3])
    v12 = v2 - v1
    v23 = v2 - v3
    return angle(v12, v23)

def distance(coor1,coor2):
    """Returns the distance between two vectors """
    val = (coor1[0]-coor2[0])*(coor1[0]-coor2[0]) \
          + (coor1[1]-coor2[1])*(coor1[1]-coor2[1]) \
          + (coor1[2]-coor2[2])*(coor1[2]-coor2[2])
    return np.sqrt(val)

def dihedral(vec1,vec2,vec3,vec4):
    """
    Returns a float value for the dihedral angle between
    the four vectors. They define the bond for which the
    torsion is calculated (~) as:
    V1 - V2 ~ V3 - V4
    The vectors vec1 .. vec4 can be array objects, lists or tuples of length
    three containing floats.
    For Scientific.geometry.Vector objects the behavior is different
    on Windows and Linux. Therefore, the latter is not a featured input type
    even though it may work.
    If the dihedral angle cant be calculated (because vectors are collinear),
    the function raises a DihedralGeometryError
    """
    # create array instances.
    v1,v2,v3,v4 = create_vectors(vec1,vec2,vec3,vec4)
    all_vecs = [v1,v2,v3,v4]

    # rule out that two of the atoms are identical
    # except the first and last, which may be.
    for i in range(len(all_vecs)-1):
        for j in range(i+1,len(all_vecs)):
            if i>0 or j<3: # exclude the (1,4) pair
                equals = all_vecs[i]==all_vecs[j]
                if equals.all():
                    raise DihedralGeometryError(\
                        "Vectors #%i and #%i may not be identical!"%(i,j))

    # calculate vectors representing bonds
    v12 = v2-v1
    v23 = v3-v2
    v34 = v4-v3

    # calculate vectors perpendicular to the bonds
    normal1 = np.cross(v12,v23)
    normal2 = np.cross(v23,v34)

    # check for linearity
    if np.linalg.norm(normal1) == 0 or np.linalg.norm(normal2)== 0:
        raise DihedralGeometryError(\
            "Vectors are in one line; cannot calculate normals!")

    # normalize them to length 1.0
    normal1 = normal1/np.linalg.norm(normal1)
    normal2 = normal2/np.linalg.norm(normal2)

    # calculate torsion and convert to degrees
    torsion = angle(normal1,normal2) * 180.0/np.pi

    # take into account the determinant
    # (the determinant is a scalar value distinguishing
    # between clockwise and counter-clockwise torsion.
    if scalar(normal1,v34) >= 0:
        return torsion
    else:
        torsion = 360-torsion
        if torsion == 360: torsion = 0.0
        return torsion

def from_cart_to_sphe(v1):
    """Convert the Cartesian vector [x, y, z] to spherical coordinates [r, theta, phi].

    The parameter r is the radial distance, theta is the polar angle, and phi is the azimuth.

    @param vector:  The Cartesian vector [x, y, z].
    @type vector:   numpy rank-1, 3D array
    @return:        The spherical coordinate vector [r, theta, phi].
    @rtype:         numpy rank-1, 3D array
    """

    # The radial distance.
    r = np.sqrt(np.sum(np.array([i*i for i in v1])))

    unit = v1/r

    # The polar angle.
    theta = atan2(unit[1],unit[0])

    # The azimuth.
    phi = atan2(np.sqrt(np.sum(np.array([i*i for i in unit[:2]]))),unit[2])

    # Return the spherical coordinate vector.
    return array([r, theta, phi], float64)

def from_cart_to_cylc(v1):
    """Convert the Cartesian vector [x, y, z] to cylindrical coordinates [r, theta, z ].

    The parameter r is the radial distance (x,y), theta is the polar angle, and z is z.

    @param vector:  The Cartesian vector [x, y, z].
    @type vector:   numpy rank-1, 3D array
    @return:        The cylindrical coordinate vector [r, theta, z].
    @rtype:         numpy rank-1, 3D array
    """

    # The radial distance.
    r = norm(v1[:1])

    # The polar angle.
    theta = atan(float(v1[1]) / v1[0])

    #
    z = v1[2]

    # Return the spherical coordinate vector.
    return array([r, theta, z], float64)

def from_sphe_to_cart(v1):
    """Convert the spherical coordinate vector [r, theta, phi] to the Cartesian vector [x, y, z].

    The parameter r is the radial distance, theta is the polar angle, and phi is the azimuth.


    @param spherical_vect:  The spherical coordinate vector [r, theta, phi].
    @type spherical_vect:   3D array or list
    @param cart_vect:       The Cartesian vector [x, y, z].
    @type cart_vect:        3D array or list
    """

    # Trig alias.
    sin_phi = sin(v1[2])

    # The vector.
    x = v1[0] * sin_phi * cos(v1[1])
    y = v1[0] * sin_phi * sin(v1[1])
    z = v1[0] * cos(v1[2])

    return array([x, y, z], float64)


def rot_x(v1,theta):
    v1 = np.matrix(v1)
    seno = np.sin(theta)
    cose = np.cos(theta)
    mtx = np.matrix([[1.,  0.  ,   0.  ],
                     [0., cose , -seno ],
                     [0., seno ,  cose ]])
    return np.array(np.sum(mtx*v1.T,axis=1).T)[0]

def rot_y(v1,theta):
    v1 = np.matrix(v1)
    seno = np.sin(theta)
    cose = np.cos(theta)
    mtx = np.matrix([[ cose   ,  0.  ,  seno  ],
                     [   0.   ,  1.  ,    0.  ],
                     [ -seno  ,  0.  ,  cose  ]])
    return np.array(np.sum(mtx*v1.T,axis=1).T)[0]

def rot_z(v1,theta):
    v1 = np.matrix(v1)
    seno = np.sin(theta)
    cose = np.cos(theta)
    mtx = np.matrix([[ cose  , -seno  ,  0. ],
                    [ seno  ,  cose,    0.  ],
                    [   0.  ,  0.  ,    1.  ]])
    return np.array(np.sum(mtx*v1.T,axis=1).T)[0]
