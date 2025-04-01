import time
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from sim import circulant_matrix

# ----------------------------------------------------------------------
# Define the MatrixFactorization module (our “model”)
# ----------------------------------------------------------------------
class MatrixFactorization(eqx.Module):
    # W_o and W_i are chosen so that W_o @ W_i has shape (L*n, L*n)
    W_o: jnp.ndarray  # shape: (L*n, hidden_size)
    W_i: jnp.ndarray  # shape: (hidden_size, L*n)

# ----------------------------------------------------------------------
# Define the loss function
# ----------------------------------------------------------------------
def loss_mf(model: MatrixFactorization, T: jnp.ndarray, L: int, n: int) -> jnp.ndarray:
    # J_L: an LxL matrix of ones
    J_L = jnp.ones((L, L))
    # Identity matrix of size n
    I_n = jnp.eye(n)
    
    # Kronecker products:
    kron_J = jnp.kron(J_L, I_n)      # shape: (L*n, L*n)
    kron_T = jnp.kron(T, I_n)          # shape: (L*n, L*n)
    
    # Compute the product.
    prod = model.W_o @ model.W_i
    # Compute the elementwise-masked difference.
    diff = (kron_J - prod) * kron_T
    # Return the squared Frobenius norm.
    return jnp.sum(diff ** 2) / jnp.sum(T)

@eqx.filter_value_and_grad
def compute_loss_mf(model: MatrixFactorization, T: jnp.ndarray, L: int, n: int) -> jnp.ndarray:
    return loss_mf(model, T, L, n)

# ----------------------------------------------------------------------
# Define a training step that computes gradients and updates the model.
# ----------------------------------------------------------------------
@eqx.filter_jit
def train_step_mf(
    model: MatrixFactorization,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    T: jnp.ndarray,
    L: int,
    n: int,
) -> tuple[jnp.ndarray, MatrixFactorization, optax.OptState]:
    loss_val, grads = compute_loss_mf(model, T, L, n)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss_val, model, opt_state

# ----------------------------------------------------------------------
# Define the training loop.
# ----------------------------------------------------------------------
def train_mf(
    key: jr.PRNGKey,
    model: MatrixFactorization,
    T: jnp.ndarray,
    L: int,
    n: int,
    optimizer_fn: callable = optax.adam,
    learning_rate: float = 1e-2,
    loss_threshold: float = 1e-6,
    max_epochs: int = 10000,
    **optimizer_kwargs,
):
    optimizer = optimizer_fn(learning_rate, **optimizer_kwargs)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    for epoch in range(max_epochs):
        loss_val, model, opt_state = train_step_mf(model, optimizer, opt_state, T, L, n)
        if epoch % (max_epochs // 10) == 0:
            print(f"Epoch {epoch}/{max_epochs}, Loss: {loss_val:.4f}")
            if loss_val < loss_threshold:
                break
    print(f"Epoch {epoch}/{max_epochs} (final), Loss: {loss_val:.4f}")
    return model

# ----------------------------------------------------------------------
# Define an experiment function to initialize and train the model.
# ----------------------------------------------------------------------
def experiment_mf(num_languages: int = 8, num_words: int = 128, hidden_size: int = 32, init_scale: float = 1e-3, seed: int = 0):
    key = jr.PRNGKey(seed)
    # Define T: a fixed binary matrix of shape (num_languages, num_languages).
    # For example, we use a lower-triangular matrix.
    # T = jnp.tril(jnp.ones((num_languages, num_languages)))
    T = circulant_matrix(num_languages, width=4)
    print("Language pairs (T):")
    print(T)
    print(f"Training on {int(jnp.sum(T)):d} ({jnp.mean(T):.2%}) language pairs.")
    L = num_languages  # L as number of languages
    n = num_words      # n as the dimension of I_n (e.g., vocabulary size)
    
    # Initialize model parameters.
    model_key1, model_key2 = jr.split(key)
    W_o = jr.normal(model_key1, (L * n, hidden_size)) * init_scale
    W_i = jr.normal(model_key2, (hidden_size, L * n)) * init_scale
    model = MatrixFactorization(W_o=W_o, W_i=W_i)
    
    initial_loss = loss_mf(model, T, L, n)
    print("Initial loss:", initial_loss)
    
    # Train the model using T.
    model = train_mf(
        key,
        model,
        T,
        L,
        n,
        optimizer_fn=optax.adam,
        learning_rate=1e-2,
        loss_threshold=1e-6,
        max_epochs=10000,
    )
    
    final_loss = loss_mf(model, T, L, n)
    print("Final training loss (using T):", final_loss)
    
    # -------------------------
    # Test Loss Computation
    # -------------------------
    # Compute the loss using 1-T instead of T.
    T_test = 1 - T
    test_loss = loss_mf(model, T_test, L, n)
    print("Test loss (using 1-T):", test_loss)
    
    # Return the model and its weights.
    return model, model.W_o, model.W_i

# ----------------------------------------------------------------------
# Running the experiment.
# ----------------------------------------------------------------------
if __name__ == '__main__':
    model, W_o, W_i = experiment_mf(num_languages=8, num_words=32, hidden_size=32, init_scale=1e-3, seed=0)
