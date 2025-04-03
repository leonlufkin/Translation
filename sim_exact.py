import time
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import os
import pandas as pd
from sim import circulant_matrix, model_l2_norm

# ----------------------------------------------------------------------
# Define the MatrixFactorization module (our “model”)
# ----------------------------------------------------------------------
class MatrixFactorization(eqx.Module):
    # W_o and W_i are chosen so that W_o @ W_i has shape (L*n, L*n)
    W_o: jnp.ndarray  # shape: (L*n, hidden_size)
    W_h: jnp.ndarray  # shape: (hidden_size, hidden_size)
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
    prod = model.W_o @ model.W_h @ model.W_i
    # prod = model.W_o @ model.W_i
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
        if epoch % 1000 == 0:
            l2norm = model_l2_norm(model)
            print(f"Epoch {epoch}/{max_epochs}, Loss: {loss_val:.4f}, L2 Norm: {l2norm:.4f}")
            if loss_val < loss_threshold:
                break
    # print(f"Epoch {epoch}/{max_epochs} (final), Loss: {loss_val:.4f}")
    return model

# ----------------------------------------------------------------------
# Define an experiment function to initialize and train the model.
# ----------------------------------------------------------------------
def experiment_mf(num_languages: int = 8, num_words: int = 128, hidden_size: int = 32, init_scale: float = 1e-3, width: int | None = None, train_frac: float | None = None, seed: int = 0):
    key = jr.PRNGKey(seed)
    dkey, mkey, ekey = jr.split(key, 3)
    
    # Define T: a fixed binary matrix of shape (num_languages, num_languages).
    assert width is not None or train_frac is not None, "Either width or train_frac must be provided."
    if width is not None:
        T = circulant_matrix(num_languages, width=width)
    elif train_frac is not None:
        num_off_diag = int((num_languages ** 2) - num_languages)
        num_train_off_diag = int((num_languages ** 2) * train_frac) - num_languages
        T = jnp.concatenate([jnp.ones(num_train_off_diag), jnp.zeros(num_off_diag - num_train_off_diag)])
        T = jr.permutation(dkey, T)
        T = T.reshape((num_languages, num_languages-1))
        T = jnp.concatenate([jnp.ones((num_languages, 1)), T], axis=1)
        # Shift each row by its index to create a circulant-like structure.
        T = jax.vmap(lambda i, row: jnp.roll(row, i))(jnp.arange(T.shape[0]), T)
        
    print("Language pairs (T):")
    print(T)
    print(f"Training on {int(jnp.sum(T)):d} ({jnp.mean(T):.2%}) language pairs.")
    L = num_languages  # L as number of languages
    n = num_words      # n as the dimension of I_n (e.g., vocabulary size)
    
    # Initialize model parameters.
    model_key1, model_key2, model_key3 = jr.split(mkey, 3)
    W_o = jr.normal(model_key1, (L * n, hidden_size)) * init_scale
    W_h = jr.normal(model_key2, (hidden_size, hidden_size)) * init_scale
    W_i = jr.normal(model_key3, (hidden_size, L * n)) * init_scale
    model = MatrixFactorization(W_o=W_o, W_h=W_h, W_i=W_i)
    
    initial_loss = loss_mf(model, T, L, n)
    print("Initial loss:", initial_loss)
    
    # Train the model using T.
    model = train_mf(
        ekey,
        model,
        T,
        L,
        n,
        optimizer_fn=optax.sgd,
        momentum=0.9,
        learning_rate=5e-3,
        loss_threshold=1e-6,
        max_epochs=500000,
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
    return test_loss, final_loss, jnp.mean(T), int(jnp.sum(T))

def main():
    # Hyperparameters.
    num_languages = 10
    num_words = 32
    hidden_size = 32
    seed = (0, 1, 2,)
    # init_scale = (1e-2,) # 1e-3, 1e-4,)
    init_scale = (1e-3, 1e-4,)
    # train_frac = (0.201, 0.211, 0.221, 0.231, 0.241, 0.251, 0.261, 0.271, 0.281, 0.291, 0.301, 0.311, 0.321, 0.331, 0.341, 0.351,)
    train_frac = (0.361, 0.371, 0.381, 0.391, 0.401, 0.411, 0.421, 0.431, 0.441, 0.451,)
    
    # Sweep through hyperparameters.
    from tqdm import tqdm
    from itertools import product
    results = []
    for idx, (s, i, p) in enumerate(tqdm(product(seed, init_scale, train_frac), desc="Running experiments")):
        print("\n\n" + 40 * "#" + f" Experiment {idx+1}/{len(train_frac) * len(init_scale) * len(seed)} " + 40 * "#")
        print(f"Running experiment with train_frac={p}, init_scale={i}, seed={s} ...\n")
        test_loss, train_loss, pair_frac, num_train = experiment_mf(
            num_languages=num_languages,
            num_words=num_words,
            hidden_size=hidden_size,
            train_frac=p,
            init_scale=i,
            seed=s,
        )
        results.append({
            "num_languages": num_languages,
            "num_words": num_words,
            "hidden_size": hidden_size,
            "train_frac": p,
            "init_scale": i,
            "seed": s,
            "test_loss": test_loss,
            "train_loss": train_loss,
        })
        # Save results.
        if os.path.exists(f"results_exact.csv"):
            # just append the most recent result at the end of the file
            df = pd.DataFrame(results[-1:], index=[0])
            df.to_csv(f"results_exact.csv", mode="a", header=False, index=False)
        else:
            df = pd.DataFrame(results)
            df.to_csv(f"results_exact.csv", index=False)
        print(f"Results saved to results_exact.csv")
        
    # Plot the results.
    plot(results=df, vocabulary_type="exact")

def plot(
    results: pd.DataFrame | None = None,
    vocabulary_type: str = "one-hot",
    truncated: int = 0,
):
    if results is None:
        results = pd.read_csv(f"results_{vocabulary_type}.csv")
    # Plot the results. Loss vs. train_frac; different lines for hidden_size; different subplots for num_words.
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="whitegrid")
    
    # Get sorted unique num_words
    unique_init_scale = sorted(results["init_scale"].unique())
    
    fig, axes = plt.subplots(1, len(unique_init_scale), figsize=(15, 5), sharey=False)
    axes = [axes] if len(unique_init_scale) == 1 else axes
    for i, init_scale in enumerate(sorted(unique_init_scale)):
        ax = axes[i]
        df = results[results["init_scale"] == init_scale]
        sns.lineplot(
            data=df,
            x="train_frac",
            y="test_loss",
            hue="seed",
            ax=ax,
            marker="o",
            markersize=5,
        )
        ax.set_title(f"init_scale={init_scale}")
        ax.set_xlabel("Fraction of training pairs")
        ax.set_ylabel("Loss")
        if truncated:
            ax.set_ylim(0, truncated)
    plt.tight_layout()
    plotname = f"results_{vocabulary_type}_truncated.pdf" if truncated else f"results_{vocabulary_type}.pdf"
    plt.savefig(plotname)
    print(f"Results saved to {plotname}")
    plt.show()

# ----------------------------------------------------------------------
# Running the experiment.
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # model, W_o, W_i = experiment_mf(
    #     num_languages=10, 
    #     num_words=32, 
    #     hidden_size=32,
    #     # hidden_size=64, 
    #     # init_scale=1e-3,
    #     init_scale=1e-4, 
    #     # width=5,
    #     # train_frac=0.30,
    #     train_frac=0.291,
    #     seed=0,
    #     # seed=1,
    # )
    # main()
    plot(vocabulary_type="exact")
    plot(vocabulary_type="exact", truncated=5)
    
# NOTE
# 1. For num_languages=10, num_words=32, hidden_size=32, init_scale=1e-3, width=5, seed=0:
#    - The model can generalize with 30% but not 29%.
#    - If I increase hidden_size to 64, the model no longer generalizes at 30%.
# 2. For num_languages=10, num_words=32, hidden_size=32, init_scale=1e-3, width=5, seed=1:
#    - The model can generalize with 27% but not 26%.