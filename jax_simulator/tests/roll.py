import jax
import jax.numpy as jnp

# Assuming vertices is your input array
vertices = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Define the rolling function
def roll_fn(shift):
    return jnp.roll(vertices, shift, axis=0)

# Define the shift values from 0 to vertices.shape[0] - 1
shift_values = jnp.arange(vertices.shape[0])

# Use jax.vmap to apply the roll function to all shift values
shifted_arrays = jax.vmap(roll_fn)(shift_values)

# shifted_arrays is now a collection of arrays, each corresponding to a shifted version of vertices
# shifted_arrays[0] corresponds to the original vertices, and shifted_arrays[i] corresponds to the vertices shifted by i positions

print(shifted_arrays)