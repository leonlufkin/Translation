import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from sim import circulant_matrix

# ----------------------------------------------------------------------
# Define the ToyModel module (our "model" for the toy case)
# ----------------------------------------------------------------------
class ToyModel(eqx.Module):
    u: jnp.ndarray  # shape: (n,)
    s: jnp.ndarray  # scalar (shape: ())
    v: jnp.ndarray  # shape: (n,)

# ----------------------------------------------------------------------
# Define the toy loss function
# ----------------------------------------------------------------------
def toy_loss(model: ToyModel, T: jnp.ndarray) -> jnp.ndarray:
    # Create an n x n matrix of ones.
    J_n = jnp.ones_like(T)
    # Compute the outer product u v^T and scale by s.
    uvT = jnp.outer(model.u, model.v)
    # Elementwise difference masked by T.
    diff = (model.s * uvT - J_n) * T
    # Return the squared Frobenius norm (with a factor of 1/2).
    return 0.5 * jnp.sum(diff ** 2)

@eqx.filter_value_and_grad
def compute_toy_loss(model: ToyModel, T: jnp.ndarray) -> jnp.ndarray:
    return toy_loss(model, T)

# ----------------------------------------------------------------------
# Define a training step that computes gradients and updates the model.
# ----------------------------------------------------------------------
@eqx.filter_jit
def train_step_toy(
    model: ToyModel,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    T: jnp.ndarray,
) -> tuple[jnp.ndarray, ToyModel, optax.OptState]:
    loss_val, grads = compute_toy_loss(model, T)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss_val, model, opt_state

# ----------------------------------------------------------------------
# Define the training loop.
# ----------------------------------------------------------------------
def train_toy(
    key: jr.PRNGKey,
    model: ToyModel,
    T: jnp.ndarray,
    optimizer_fn: callable = optax.sgd,
    learning_rate: float = 1e-2,
    loss_threshold: float = 1e-6,
    max_steps: int = 10000,
    **optimizer_kwargs,
) -> ToyModel:
    optimizer = optimizer_fn(learning_rate=learning_rate, **optimizer_kwargs)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    for step in range(max_steps):
        loss_val, model, opt_state = train_step_toy(model, optimizer, opt_state, T)
        if step % 1000 == 0:
            print(f"Step {step}/{max_steps}, Loss: {loss_val:.6f}")
            if loss_val < loss_threshold:
                break
    return model

# ----------------------------------------------------------------------
# Define an experiment function to initialize and train the toy model.
# ----------------------------------------------------------------------
def experiment_toy(n: int = 32, init_scale: float = 1e-3, p: float = 0.5, seed: int = 0):
    key = jr.PRNGKey(seed)
    
    # Create a binary matrix T of shape (n, n).
    # Here, we set the diagonal to 1 and sample off-diagonals from a Bernoulli with probability p.
    # T = jr.bernoulli(key, p, shape=(n, n)).astype(jnp.float32)
    # T = T.at[jnp.diag_indices(n)].set(1.0)
    T = circulant_matrix(n, 2)
    
    print("Binary mask T:")
    print(T)
    print(f"Fraction of ones in T: {jnp.mean(T) * 100:.1f}%")
    U, S, V = jnp.linalg.svd(T)
    V = V.T
    print("SVD of T:")
    print("U[:,0]:", U[:, 0])
    print("S:", S)
    print("V[:,0]:", V[:, 0])
    breakpoint()
    
    # Initialize model parameters.
    key, subkey1, subkey2, subkey3 = jr.split(key, 4)
    u = jr.normal(subkey1, (n,)) * init_scale
    s = jr.normal(subkey2, ()) * init_scale  # scalar parameter
    v = jr.normal(subkey3, (n,)) * init_scale
    model = ToyModel(u=u, s=s, v=v)
    
    initial_loss = toy_loss(model, T)
    print("Initial loss:", initial_loss)
    
    # Train the model using the binary mask T.
    model = train_toy(
        key,
        model,
        T,
        optimizer_fn=optax.sgd,
        learning_rate=5e-3,
        loss_threshold=1e-6,
        momentum=0.9,
        max_steps=5000000,
    )
    
    final_loss = toy_loss(model, T)
    print("Final loss:", final_loss)
    test_loss = toy_loss(model, 1 - T)
    print("Test loss:", test_loss)
    
    return model, final_loss, test_loss, T

# ----------------------------------------------------------------------
# Main entry point.
# ----------------------------------------------------------------------
def main():
    model, final_loss, test_loss, T =  experiment_toy(n=10, init_scale=1e-5, p=0.0, seed=0)
    print("Final model parameters:")
    print("u:", model.u)
    print("s:", model.s)
    print("v:", model.v)
    U, S, V = jnp.linalg.svd(T)
    V = V.T
    print("SVD of T:")
    print("U[:,0]:", U[:, 0])
    print("S:", S)
    print("V[:,0]:", V[:, 0])
    breakpoint()

if __name__ == '__main__':
    main()
