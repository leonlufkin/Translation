import time

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from collections.abc import Generator
from jax import Array

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxnets import datasets, models, samplers
from jaxnets import utils as jnutils
import equinox as eqx
import optax
import pandas as pd
import os

from utils import mse, batcher

def circulant_matrix(n, width):
    # Create the first row: 'fatness' ones followed by zeros.
    first_row = jnp.concatenate([jnp.ones(width), jnp.zeros(n - width)])
    # Build the circulant matrix by circularly shifting the first row.
    return jnp.stack([jnp.roll(first_row, i) for i in range(n)])

def model_l2_norm(model):
    # Extract only the array leaves from the model.
    weights = eqx.filter(model, eqx.is_array)
    # Flatten the pytree into a list of arrays.
    leaves = jax.tree_util.tree_leaves(weights)
    # Compute the squared L2 norm of each leaf and sum them up.
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))

class LanguagePairDataset(datasets.Dataset):
  """A `Dataset` of finite class exemplars from which to draw sequences."""
  
  num_exemplars: int
  num_dimensions: int
  sequence_length: int
  words: Array
  pairs: Array
  
  def __init__(
    self,
    key: Array,
    words: Array,
    pairs: Array,
  ):
    """A `Dataset` of finite class exemplars from which to draw examples.

    Args:
      key: A key for randomness in sampling.
      inputs: A sequence of inputs.
      labels: A sequence of labels.
    """
    super().__init__(
      key=key,
      num_exemplars=len(words) * len(pairs),
      num_dimensions=words.shape[-1] + pairs.shape[-1],
      sequence_length=0,
    )
    self.words = words
    self.pairs = pairs
    
  def __getitem__(self, index: int | slice):
    """Return an exemplar from the dataset."""
    if isinstance(index, (int, slice)):
      i = index
      i = jnutils.slice_to_array(i, len(self)) if isinstance(i, slice) else jnp.array([i])
      words_index = i // len(self.words)
      pairs_index = i % len(self.pairs)
      if isinstance(index, int):
        words_index, pairs_index = words_index.item(), pairs_index.item()
      return (self.words[words_index], self.pairs[pairs_index]), self.words[words_index]
    raise TypeError(f"Index type {type(index)} not supported.")

class SharedPathwayNet(models.Net):
  """
  A gated deep linear network implementation of a shared pathways model.
  """
  
  input_layers: list[eqx.Module]
  hidden_layer: eqx.Module
  output_layers: list[eqx.Module]
  in_gates: list[Callable]
  out_gates: list[Callable]

  def __init__(
    self,
    in_size: int,
    hidden_size: int,
    out_size: int,
    in_gates: list[Callable],
    out_gates: list[Callable],
    *,
    key: Array = None,
    init_fn: Callable = models.xavier_normal_init,
    **linear_kwargs
  ):
    super().__init__()

    num_input_layers = len(in_gates)
    num_output_layers = len(out_gates)
    self.in_gates = in_gates
    self.out_gates = out_gates
    
    self.input_layers = tuple(
      models.Linear(
        in_size=in_size,
        out_size=hidden_size,
        key=jr.fold_in(key, i),
        init_fn=init_fn,
        **linear_kwargs,
      )
      for i in range(num_input_layers)
    )
    self.hidden_layer = models.Linear(
      in_size=hidden_size,
      out_size=hidden_size,
      key=jr.fold_in(key, num_input_layers),
      init_fn=init_fn,
      **linear_kwargs,
    )
    self.output_layers = tuple(
      models.Linear(
        in_size=hidden_size,
        out_size=out_size,
        key=jr.fold_in(key, i),
        init_fn=init_fn,
        **linear_kwargs,
      )
      for i in range(num_input_layers + 1, num_input_layers + num_output_layers + 1) # Need to make sure we have different random seeds from the input layers.
    )
    
  def forward_pass(self, x: Array, *, key: Array) -> Array:
    """Apply the MLP block to the input."""
    w, p = x # w: "word" (input), p: "language pair" (determines )
    # 1. Input layers.
    a1 = jnp.array([ layer(w) for layer in self.input_layers ])
    g1 = jnp.expand_dims(jnp.array([ gate(p) for gate in self.in_gates ]), axis=1)
    y1 = jnp.sum(a1 * g1, axis=0)
    # 2. Hidden layer. 
    y2 = self.hidden_layer(y1)
    # 3. Output layers.
    a2 = jnp.array([ layer(y2) for layer in self.output_layers ])
    g2 = jnp.expand_dims(jnp.array([ gate(p) for gate in self.out_gates ]), axis=1)
    z = jnp.sum(a2 * g2, axis=0)
    # 3. Final output.
    return z, a1, g1, y1, y2, a2, g2
    
    
@eqx.filter_value_and_grad
def compute_loss(model: eqx.Module, x: Array, y: Array, key: Array, loss_fn: Callable) -> Array:
    """Compute cross-entropy loss on a single example."""
    keys = jr.split(key, x[0].shape[0] if type(x) is tuple else x.shape[0])
    pred_y = jax.vmap(model)(x, key=keys)
    loss = loss_fn(pred_y, y)
    return loss.mean()
        
@eqx.filter_jit
def train_step(
    model: eqx.Module,
    optimizer: optax.GradientTransformation,
    opt_state: Array,
    x: Array,
    y: Array,
    key: Array,
    loss_fn: Callable,
) -> tuple[Array, eqx.Module, Array]:
    """Train the model on a single example."""
    loss, grads = compute_loss(model, x, y, key, loss_fn)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

def train(
    key: jr.PRNGKey,
    model: eqx.Module,
    dataset: datasets.Dataset,
    # sampler: samplers.Sampler, # We will use samplers.DirectSampler for now
    loss_fn: Callable = mse,
    optimizer_fn: Callable = optax.sgd,
    learning_rate: float = 1e-3,
    loss_threshold: float = 1e-3,
    max_epochs: int = 100,
    **optimizer_kwargs
):
    # Setup the sampler.
    sampler = samplers.DirectSampler(dataset)
    print(f"Length of sampler: {len(sampler)}")
    
    # Initialize the optimizer.
    optimizer = optimizer_fn(learning_rate, **optimizer_kwargs)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Training.
    print("\nStarting training...")
    start_time = time.time()
    for epoch in range(max_epochs):
        x, y = dataset[:]
        loss, model, opt_state = train_step(
            model, optimizer, opt_state, x, y, key, loss_fn
        )
        if epoch % int(max_epochs//10) == 0:
            l2norm = model_l2_norm(model)
            print(f"Epoch {epoch}/{max_epochs}, Loss: {loss:.4f}, L2 norm: {l2norm:.4f}")
            if loss < loss_threshold:
                break
    end_time = time.time()
    print(f"Epoch {epoch}/{max_epochs} (final) loss: {loss:.4f}")
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    return model
    

def experiment(
    # Hyperparameters.
    num_languages = 8,
    num_words = 128,
    hidden_size = 32,
    circulant_width = 4,
    init_scale = 1e-3,
    seed = 0,
    vocabulary_type = "one-hot",
):
    # Randomness.
    key = jr.PRNGKey(seed)
    mkey, dkey, skey = jr.split(key, 3)
    
    # Define a simple MLP model.
    lnn = SharedPathwayNet(
        in_size=num_words,
        hidden_size=hidden_size,
        out_size=num_words,
        in_gates=[lambda x, i=i: x[0] == i for i in range(num_languages)],
        out_gates=[lambda x, i=i: x[1] == i for i in range(num_languages)],
        key=mkey,
        use_bias=False,
        init_scale=init_scale ** 2,
        # add initialization scale here
    )
    print("L2 norm of weights at initialization:", model_l2_norm(lnn))
    
    # Define the simple dataset.
    # 1. Define the language pairs.
    language_matrix = circulant_matrix(num_languages, circulant_width)
    print("Language matrix:")
    print(language_matrix)
    input_language, output_language = jnp.nonzero(language_matrix)
    train_language_pairs = jnp.array([input_language, output_language]).T
    pair_frac = len(train_language_pairs)/num_languages**2
    print(f"Training on {len(train_language_pairs)}/{num_languages**2} ({pair_frac:.2%}) language pairs.")
    # 2. Define the vocabulary (shared across languages, for now).
    if not vocabulary_type in ["one-hot", "gaussian"]:
      raise ValueError(f"Unknown vocabulary type: {vocabulary_type}")
    vocabulary = jnp.eye(num_words) if vocabulary_type == "one-hot" else \
      jr.normal(mkey, (1000, num_words))
    # 3. Define the dataset.
    train_dataset = LanguagePairDataset(
        key=dkey,
        words=vocabulary,
        pairs=train_language_pairs,
    )
    # # 3.5. Test the model & dataset.
    # x, yy = train_dataset[0]
    # z, a1, g1, y, a2, g2 = lnn.forward_pass(x, key=skey)
    # 4. Train the model.
    final_lnn = train(
        key=dkey,
        model=lnn,
        dataset=train_dataset,
        # optimizer_fn=optax.adam,
        optimizer_fn=optax.sgd,#sgd,
        momentum=0.9,
        learning_rate=5e-2,
        loss_threshold=1e-6,
        max_epochs=10000,
    )
    # Print l2 norm of weights
    print("L2 norm of weights after training:", model_l2_norm(final_lnn))
    # 5. Create the holdout dataset.
    input_language, output_language = jnp.nonzero(1 - language_matrix)
    test_language_pairs = jnp.array([input_language, output_language]).T
    test_dataset = LanguagePairDataset(
        key=dkey,
        words=vocabulary,
        pairs=test_language_pairs,
    )
    # 5. Test the model on the holdout language pairs.
    x, y = test_dataset[:]
    z = jax.vmap(final_lnn)(x, key=jr.split(skey, len(test_dataset)))
    loss = jnp.square(z - y).sum(axis=-1).mean()
    accuracy = jnp.mean(jnp.argmax(z, axis=-1) == jnp.argmax(y, axis=-1))
    print(f"Loss on holdout dataset: {loss:.4f}, Accuracy: {accuracy:.2%}")
    # # 5.5. Look at an example from the test dataset.
    # x, yy = test_dataset[0]
    # z, a1, g1, y, a2, g2 = final_lnn.forward_pass(x, key=skey)
    # breakpoint()
    # 6. Save results.
    return loss, accuracy, pair_frac, len(train_dataset)

def main():
    # Hyperparameters.
    num_languages = 10
    num_words = (128,) # (8, 16, 24, 32, 128,)
    hidden_size = (64,) # (4, 32, 128,)
    circulant_width = (1, 2, 3, 4,)
    init_scale = 0.2 # 1e-3
    seed = 0
    vocabulary_type = "one-hot" # "one-hot", "gaussian"
    
    # Sweep through hyperparameters.
    from tqdm import tqdm
    from itertools import product
    results = []
    for i, (n, h, w) in enumerate(product(num_words, hidden_size, circulant_width)):
        print("\n\n" + 40 * "#" + f" Experiment {i+1}/{len(num_words) * len(hidden_size) * len(circulant_width)} " + 40 * "#")
        print(f"Running experiment with num_words={n}, hidden_size={h}, circulant_width={w}, vocabulary_type={vocabulary_type} ...\n")
        loss, accuracy, pair_frac, num_train = experiment(
            num_languages=num_languages,
            num_words=n,
            hidden_size=h,
            circulant_width=w,
            init_scale=init_scale,
            seed=seed,
            vocabulary_type=vocabulary_type,
        )
        results.append({
            "num_languages": num_languages,
            "num_words": n,
            "hidden_size": h,
            "circulant_width": w,
            "loss": loss,
            "accuracy": accuracy,
            "pair_frac": pair_frac,
            "num_train": num_train,
        })
        # Save results.
        if os.path.exists(f"results_{vocabulary_type}.csv"):
            # just append the most recent result at the end of the file
            df = pd.DataFrame(results[-1:], index=[0])
            df.to_csv(f"results_{vocabulary_type}.csv", mode="a", header=False, index=False)
        else:
            df = pd.DataFrame(results)
            df.to_csv(f"results_{vocabulary_type}.csv", index=False)
        print(f"Results saved to results_{vocabulary_type}.csv")
    # Plot the results.
    plot(df, vocabulary_type=vocabulary_type)

def plot(
    results: pd.DataFrame | None = None,
    vocabulary_type: str = "one-hot",
):
    if results is None:
        results = pd.read_csv(f"results_{vocabulary_type}.csv")
    # Plot the results. Loss vs. train_frac; different lines for hidden_size; different subplots for num_words.
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="whitegrid")
    
    # Get sorted unique num_words
    unique_num_words = sorted(results["num_words"].unique())
    
    fig, axes = plt.subplots(1, len(unique_num_words), figsize=(15, 5), sharey=False)
    for i, num_words in enumerate(unique_num_words):
        ax = axes[i]
        df = results[results["num_words"] == num_words]
        sns.lineplot(
            data=df,
            x="pair_frac",
            y="loss",
            hue="hidden_size",
            ax=ax,
            marker="o",
            markersize=5,
        )
        ax.set_title(f"num_words={num_words}")
        ax.set_xlabel("Fraction of training pairs")
        ax.set_ylabel("Loss")
    plt.tight_layout()
    plt.savefig(f"results_{vocabulary_type}.png")
    print(f"Results saved to results_{vocabulary_type}.png")
    plt.show()

    

if __name__ == '__main__':
    main()