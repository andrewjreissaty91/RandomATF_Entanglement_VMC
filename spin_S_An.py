import argparse
import os
import sys
from transformer_LR_WF.autoregressive_An import *
import netket as nk
from netket.operator.spin import sigmax,sigmaz 
nk.config.netket_spin_ordering_warning = False  
import jax.numpy as jnp
import jax
from jax import random
import numpy as np
from netket import jax as nkjax
import netket.operator as op
from functools import partial
from netket.vqs import MCState
from netket.stats import statistics as mpi_statistics
from flax import linen as nn
argument_parser = argparse.ArgumentParser(
    description="run an example calculation for a spin model"
)

argument_parser.add_argument(
    "N", type=int, help="number of spins in the chain"
)

argument_parser.add_argument(
    "J", type=float, help="value of the base spin-spin interaction strength"
)
argument_parser.add_argument(
    "h", type=float, help="spin strength"
)

argument_parser.add_argument(
    "num_layers", type=int, help="num_layers"
)

argument_parser.add_argument(
    "num_heads", type=int, help="number of head"
)
argument_parser.add_argument(
    "dff", type=int, help="dff"
)
argument_parser.add_argument(
    "d_model", type=int, help="embedding"
)
argument_parser.add_argument(
    "embedding_var", type=float, help="variance of parameter"
)
argument_parser.add_argument(
    "positive_or_complex", type=str, help="positive or complex"
)
argument_parser.add_argument(
    "softmaxOnOrOff", type=str, help="softmaxOnOrOff"
)
argument_parser.add_argument(
    "modulusOnOrOff", type=str, help="modulusOnOrOff"
)
argument_parser.add_argument(
    "signOfProbsIntoPhase", type=str, help="signOfProbsIntoPhase"
)
argument_parser.add_argument(
    "ARD_vs_MCMC", type=str, help="Sampler"
)
argument_parser.add_argument(
    "run", type=int, help="run"
)
argument_parser.add_argument(
    "attn_mode", type=str, help="attention_mode"
)
argument_parser.add_argument(
    "activation", type=str, help="activation function: ReLU or I"
)

argument_parser.add_argument(
    "Quantity", type=str, help="Quantity to calculate"
)

args = argument_parser.parse_args()

N = args.N
J = args.J
h = args.h
num_layers = args.num_layers
run = args.run
num_heads = args.num_heads
dff = args.dff  
d_model = int(args.d_model)
embedding_var = args.embedding_var
positive_or_complex = args.positive_or_complex
softmaxOnOrOff = args.softmaxOnOrOff
modulusOnOrOff = args.modulusOnOrOff
signOfProbsIntoPhase = args.signOfProbsIntoPhase
ARD_vs_MCMC = args.ARD_vs_MCMC
attn_mode= args.attn_mode
activation_name = args.activation
Quantity = args.Quantity

if activation_name.lower() == "relu":
    activation_fn = nn.relu
elif activation_name.lower() == "i":
    activation_fn = lambda x: x
else:
    raise ValueError(f"Unsupported activation function: {activation_name}")

hi = nk.hilbert.Spin(s=0.5, N=N) 
model = Transformer(hilbert=hi,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    d_model=d_model,
                    dff=dff, autoreg=True,
                    embedding_var=embedding_var,
                    positive_or_complex=positive_or_complex,
                    softmaxOnOrOff=softmaxOnOrOff,
                    modulusOnOrOff=modulusOnOrOff,
                    signOfProbsIntoPhase=signOfProbsIntoPhase,
                    attn_mode=attn_mode,
                    activation_fn=activation_fn
                    )

### MC sampling rules ###
if ARD_vs_MCMC == "ARD":
    sampler = nk.sampler.ARDirectSampler(hi)
    vstate = nk.vqs.MCState(sampler=sampler, model=model, n_samples=102400)

elif ARD_vs_MCMC == "MCMC":
    sampler = nk.sampler.MetropolisLocal(hi,n_chains=512,sweep_size=128)
    vstate = nk.vqs.MCState(sampler=sampler, model=model, n_samples=102400,n_discard_per_chain=100, chunk_size=1024)

  
SEED = int(N * d_model* 100000 + run*100)
vstate.init(SEED)

### Training schedule ###
max_iters = 0
optimizer = nk.optimizer.Adam(learning_rate=0.001)
H = sum (J * sigmaz(hi, i) * sigmaz(hi, (i+1)%N) for i in range(N-1) ) +  sum(h* sigmax(hi, i) for i in range(N)) 
gs = nk.driver.VMC(
    H,
    optimizer,
    variational_state=vstate)
gs.run(n_iter=max_iters, show_progress=False)

if Quantity == "Ent":
    N_s = 204800  # Number of samples
    samples = vstate.sample(n_samples=N_s)
    n_samples = samples.shape[0] * samples.shape[1]
    n_chains = samples.shape[0]
    if n_chains == 1: 
        samples_1 = samples[:, : (n_samples // 2)]
        samples_2 = samples[:, (n_samples // 2):]
    else:
        samples_1 = samples[:(n_chains // 2)]
        samples_2 = samples[(n_chains // 2):]

    # Define swap and original states 
    @partial(jax.jit, static_argnames=("afun", "chunk_size"))
    def Renyi_2_entropy(afun, params, model_state, samples_1, samples_2, partition, *, chunk_size):
        sigma_A = samples_1.reshape(-1, N)
        sigma_B = samples_2.reshape(-1, N)
        # Swap states
        sigma_A_tilde = jnp.copy(sigma_A)
        sigma_B_tilde = jnp.copy(sigma_B)
        sigma_A_tilde = sigma_A_tilde.at[:, partition].set(sigma_B[:, partition])
        sigma_B_tilde = sigma_B_tilde.at[:, partition].set(sigma_A[:, partition])
        
        @partial(nkjax.apply_chunked, in_axes=(None, None, 0, 0, 0, 0), chunk_size=chunk_size)
        def compute_entropy(params, model_state, sigma_A, sigma_B, sigma_A_tilde, sigma_B_tilde):
            W = {"params": params, **model_state}
            return jnp.exp(
                afun(W, sigma_A_tilde) + afun(W, sigma_B_tilde) - afun(W, sigma_A) - afun(W, sigma_B)
            )
        # Call the inner function and return the result
        return compute_entropy(params, model_state, sigma_A, sigma_B, sigma_A_tilde, sigma_B_tilde)

    # Calculate for the half-system
    half_system_index = N // 2 - 1  # Index for L_A = N/2 (0-based indexing)
    half_system_partition = jnp.arange(N // 2)
    Renyi_2_half_system = Renyi_2_entropy(
        vstate._apply_fun, vstate.parameters, vstate.model_state, 
        samples_1, samples_2, partition=half_system_partition, chunk_size=1024
    )

    # Calculate statistics for the half-system
    Renyi2_half_stats = mpi_statistics(Renyi_2_half_system.reshape((n_chains, -1)).T) 
    Renyi2_half_mean_ln = -jnp.log(Renyi2_half_stats.mean).real 
    Renyi2_half_error = float(jnp.sqrt(Renyi2_half_stats.variance / N_s) / Renyi2_half_stats.mean.real)
    # Print half-system results
    print(Renyi2_half_mean_ln, Renyi2_half_error)

elif Quantity == "Cor":
    # Define Sz operator
    def Sz(hilbert, i):
        return op.spin.sigmaz(hilbert, i)

    # Function to compute C_zz for a given pair (i, j)
    def compute_C_zz(vstate, i, j):
        hilbert = vstate.hilbert
        Sz_i = Sz(hilbert, i)
        Sz_j = Sz(hilbert, j)
        
        # Compute expectation values
        Sz_i_Sz_j = Sz_i @ Sz_j
        expect_Sz_i_Sz_j = vstate.expect(Sz_i_Sz_j).mean
        expect_Sz_i = vstate.expect(Sz_i).mean
        expect_Sz_j = vstate.expect(Sz_j).mean
        
        # Correlation C_zz
        C_zz = expect_Sz_i_Sz_j - expect_Sz_i * expect_Sz_j
        return C_zz


    # Compute C_zz for different distances
    distances = range(1, N // 2 + 1)
    C_zz_avg = []
    for d in distances:
        C_zz_vals = []
        for i in range(N):
            j = (i + d) % N
            C_zz = compute_C_zz(vstate, i, j)
            abs_C_zz = jnp.abs(C_zz)
            if abs_C_zz == 0 or jnp.isnan(abs_C_zz):
                sys.exit()
            C_zz_vals.append(abs_C_zz)

        if C_zz_vals:
            C_zz_avg.append(jnp.log(np.mean(C_zz_vals)))

    # Print the averages on a single line
    print(' '.join(map(str, np.array(C_zz_avg).real)))

elif Quantity == "Sch":
    configs = []
    import itertools as it
    import string
    def partial_trace_np(psi: np.ndarray, keep: list, dims: list) -> np.ndarray:
        r"""
        Calculate the partial trace of an outer product

        .. math::

        \rho_a = \text{Tr}_b (| u \rangle \langle u |)

        Args:
            *psi (tensor)*:
                Quantum state of shape (None ,2,2,...,2), where None is a batch dimension.

            *keep (list)*:
                An array of indices of the spaces to keep after being traced. For instance, if the space is
                A x B x C x D and we want to trace out B and D, keep = [0,2]

            *dims (list)*:
                An array of the dimensions of each space. For instance, if the space is A x B x C x D,
                dims = [None, dim_A, dim_B, dim_C, dim_D]. None is used as a batch dimension.

        Returns (Tensor):
            Partially traced out matrix

        """
        letters = string.ascii_lowercase + string.ascii_uppercase
        keep = [k + 1 for k in keep]
        assert 2 * max(keep) < len(letters) - 1, "Not enough letters for einsum..."
        keep = np.asarray(keep)
        dims = np.asarray(dims)
        Ndim = dims.size
        Nkeep = np.prod(dims[keep])

        idx1 = letters[-1] + ''.join([letters[i] for i in range(1, Ndim)])
        idx2 = letters[-1] + ''.join([letters[Ndim + i] if i in keep else letters[i] for i in range(1, Ndim)])
        idx_out = letters[-1] + ''.join([i for i, j in zip(idx1, idx2) if i != j] + [j for i, j in zip(idx1, idx2) if i != j])
        psi = np.reshape(psi, dims)
        rho_a = np.einsum(idx1 + ',' + idx2 + '->' + idx_out, psi, np.conj(psi))
        return np.reshape(rho_a, (-1, Nkeep, Nkeep))
    for k in range(N):
        configs.append((-1, 1))
    configs = list(it.product(*configs))  
    configs_full = np.array(configs)
    eigenvalues_rho_A = []
    shape = tuple(np.full(N, 2))
    shape = tuple(np.append(1, shape))
    parameters = vstate.parameters
    log_psi = model.apply({'params': parameters}, configs_full)
    psi = jnp.exp(log_psi)
    if ARD_vs_MCMC == 'MCMC':
        norm_sq = jnp.sum(psi * jnp.conj(psi))
        norm = jnp.sqrt(norm_sq)
        psi = psi / norm
    psi = np.reshape(psi, shape)
    numSpins_A = N // 2
    keep = list(np.arange(0, numSpins_A))
    dims = list(np.full(N, 2))
    dims = [1] + dims
    rho_A = partial_trace_np(psi=psi, keep=keep, dims=dims)[0]
    rho_A = rho_A / np.trace(rho_A)
    eigenvalues, eigenvectors = jax.scipy.linalg.eigh(rho_A)
    eigenvalues = np.clip(eigenvalues, 0, None)
    print(" ".join(map(str, eigenvalues)))

