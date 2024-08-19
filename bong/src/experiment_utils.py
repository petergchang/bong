import jax.numpy as jnp
import jax.random as jr


def generate_rotation_matrix(key, d):
    # Generate a random orthogonal matrix
    R = jr.orthogonal(key, d)
    # Check if the determinant is -1
    if jnp.linalg.det(R) < 0:
        # Multiply the first column by -1 to flip the determinant to 1
        R = R.at[:, 0].multiply(-1)
    return R


def generate_covariance_matrix(key, d, c, scale):
    vec = (1 / jnp.arange(1, d + 1) ** c) * scale**2
    vec = vec / jnp.linalg.norm(vec)
    cov_u = jnp.diag(vec)
    Q = generate_rotation_matrix(key, d)
    cov = Q.T @ cov_u @ Q
    return cov
