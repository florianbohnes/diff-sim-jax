import jax.numpy as jnp
from jax import lax
from jax import jit
import jax

@jit
def perpendicular(pt):
    """Return a 2-vector's perpendicular vector.

    Args:
        pt (jnp.ndarray, (2,)): input vector

    Returns:
        pt (jnp.ndarray, (2,)): perpendicular vector
    """
    temp = pt[0]
    pt = jnp.array([pt[1], -1 * temp])
    return pt

@jit
def tripleProduct(a, b, c):
    """Return triple product of three vectors.

    Args:
        a, b, c (jnp.ndarray, (2,)): input vectors

    Returns:
        (jnp.ndarray, (2,)): triple product
    """
    ac = a.dot(c)
    bc = b.dot(c)
    return b * ac - a * bc

@jit
def indexOfFurthestPoint(vertices, d):
    """Return the index of the vertex furthest away along a direction in the list of vertices.

    Args:
        vertices (jnp.ndarray, (n, 2)): the vertices we want to find avg on

    Returns:
        idx (int): index of the furthest point
    """
    return jnp.argmax(vertices.dot(d))

@jit
def support(vertices1, vertices2, d):
    """Return the support point for the Minkowski Difference of two shapes in a given direction.

    Args:
        vertices1 (jnp.ndarray, (n, 2)): vertices of the first shape
        vertices2 (jnp.ndarray, (n, 2)): vertices of the second shape
        d (jnp.ndarray, (2,)): direction

    Returns:
        support (jnp.ndarray, (2,)): support point
    """
    i = indexOfFurthestPoint(vertices1, d)
    j = indexOfFurthestPoint(vertices2, -d)
    return vertices1[i] - vertices2[j]

@jit
def avgPoint(vertices):
    """Return the average point of multiple vertices.

    Args:
        vertices (jnp.ndarray, (n, 2)): the vertices we want to find avg on

    Returns:
        avg (jnp.ndarray, (2,)): average point of the vertices
    """
    return jnp.sum(vertices, axis=0) / vertices.shape[0]

@jit
def collision(vertices1, vertices2):
    """Check if two shapes are colliding.

    Args:
        vertices1 (jnp.ndarray, (n, 2)): vertices of the first shape
        vertices2 (jnp.ndarray, (n, 2)): vertices of the second shape

    Returns:
        colliding (bool): True if colliding, False otherwise
    """
    index = 0
    simplex = jnp.empty((3, 2))
    position1 = avgPoint(vertices1)
    position2 = avgPoint(vertices2)

    d = position1 - position2

    d = d.at[0].set(lax.cond(
        d[0] == 0,
        lambda _: lax.cond(d[1] > 0, lambda _: 1.,
                 lambda _: d[0],
                 operand=None),
        lambda _: d[0],
        operand=None
    ))

    a = support(vertices1, vertices2, d)

    simplex = simplex.at[index, :].set(a)


    collision = lax.cond(d.dot(a) <= 0,
                            lambda _: False,
                            lambda _: True,
                            operand=None)

    d = -a

    def loop_body(carry, _):
        simplex, index, d = carry
        index += 1
        index = index.astype(int)
        a = support(vertices1, vertices2, d)
        simplex = simplex.at[index, :].set(a)

        # if throw here
        collision = lax.cond(d.dot(a) <= 0,
                            lambda _: False,
                            lambda _: True,
                            operand=None)


        ao = -a

        d = lax.cond(index < 2,
                     lambda args: get_d(*args),
                     lambda args: args[3],  # Return 'd' unchanged when index >= 2
                     (simplex, ao, a, d))

        c = simplex[0, :]
        b = simplex[1, :]
        ab = b - a
        ac = c - a

        acperp = tripleProduct(ab, ac, ac)

        d, collision = lax.cond(acperp.dot(a) >= 0,
                                lambda _: (acperp, collision),
                                lambda _: abperp_logic(*_),
                                (ac, ab, ao))

        simplex = simplex.at[1].set(simplex[2])
        index -= 1

        return carry, collision

    # initialise loop variables
    index = 0
    loop_carry = (simplex, index, d)
    # loop(scan) is performed 1000 times
    _, collision = lax.scan(loop_body, loop_carry, jnp.arange(1000))

    # Scan function returns array with boolean entries of every iteration
    # flatten to only one entry
    collision = jnp.any(collision)

    return collision




@jit
def get_d(simplex, a, ao, d):
    '''
    Helper function carrying some logic that would be inside if statement
    if this was regular python.
    Args:
        simplex:
        a:
        ao:
        d:

    Returns:
        d
    '''
    b = simplex[0, :]
    ab = b - a
    d = tripleProduct(ab, ao, ab)
    d = lax.cond(jnp.linalg.norm(d) < 1e-10,
                 lambda _: perpendicular(ab),
                 lambda _: d,
                 operand=None)
    return d

@jit
def abperp_logic(ac, ab, ao):
    """
    This is again a helper function that would be inside an if statement in regular python
    """
    abperp = tripleProduct(ac, ab, ab)
    collision = lax.cond(abperp.dot(ao) < 0,
                            lambda _: True,
                            lambda _: False,
                            operand=None)

    return abperp, collision


@jit
def collision_multiple(vertices):
    """Check pair-wise collisions for all provided vertices.

    Args:
        vertices (np.ndarray (num_bodies, 4, 2)): all vertices for checking pair-wise collision

    Returns:
        collisions (np.ndarray (num_vertices, )): whether each body is in collision
   """

    global COLLISION_REF
    COLLISION_REF = collision
    # this is so the function 'collision' will be available withing lax.scan below
    # collision takes as arguments two sets of vertices

    def scan_vertices_outer(carry, current_vertex):
        carry_outer = current_vertex


        def scan_vertices_inner(carry, current_vertex):
            global COLLISION_REF
            collision = COLLISION_REF(carry_outer, current_vertex)
            # call collision function from outside here
            carry_inner = carry_outer
            return carry_inner, collision

        first_vertex = vertices[0]
        # carry value takes first_index as init
        what, collision = lax.scan(scan_vertices_inner, init=first_vertex, xs=vertices)

        return carry, collision

    # What would be a nested loop in regular python (to compare each vertex to every other vertex)
    # is done here by nesting lax.scan functions
    first_vertex = vertices[0]
    _, collisions = lax.scan(scan_vertices_outer, init=first_vertex, xs=vertices)


    #rows, cols = jnp.where(collisions)
    #collision_idx = -jnp.ones(collisions.shape[0], dtype=int)
    #collision_idx[rows] = cols
    #collision_idx = collision_idx.at[rows].set(cols)

    # flatten collisions
    collisions = jnp.any(collisions, axis=1)
    return collisions




@jit
def get_trmtx(pose):
    """Get transformation matrix of vehicle frame -> global frame,

        Args:
            pose (np.ndarray (3, )): current pose of the vehicle

        return:
            H (np.ndarray (4, 4)): transformation matrix
    """
    x = pose[0]
    y = pose[1]
    th = pose[2]
    cos = jnp.cos(th)
    sin = jnp.sin(th)
    H = jnp.array([[cos, -sin, 0., x], [sin, cos, 0., y], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    return H

@jit
def get_vertices(pose, length, width):
    """Utility function to return vertices of the car body given pose and size.

    Args:
        pose (np.ndarray, (3, )): current world coordinate pose of the vehicle
        length (float): car length
        width (float): car width

    Returns:
        vertices (np.ndarray, (4, 2)): corner vertices of the vehicle body
    """
    H = get_trmtx(pose)
    rl = H.dot(jnp.asarray([[-length / 2], [width / 2], [0.], [1.]])).flatten()
    rr = H.dot(jnp.asarray([[-length / 2], [-width / 2], [0.], [1.]])).flatten()
    fl = H.dot(jnp.asarray([[length / 2], [width / 2], [0.], [1.]])).flatten()
    fr = H.dot(jnp.asarray([[length / 2], [-width / 2], [0.], [1.]])).flatten()
    rl = rl / rl[3]
    rr = rr / rr[3]
    fl = fl / fl[3]
    fr = fr / fr[3]
    vertices = jnp.asarray([[rl[0], rl[1]], [rr[0], rr[1]], [fr[0], fr[1]], [fl[0], fl[1]]])
    return vertices

