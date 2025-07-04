# Copyright 2024 the authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import pathlib
from typing import Optional,Any, Callable, Tuple
import flax
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from netket.sampler import MetropolisRule
from netket.utils.struct import dataclass

Array = Any

REAL_DTYPE = jnp.asarray(1.0).dtype


def circulant(
    row: npt.ArrayLike, times: Optional[int] = None
) -> npt.ArrayLike:
    """Build a (full or partial) circulant matrix based on an array.

    Args:
        row: The first row of the matrix.
        times: If not None, the number of rows to generate.

    Returns:
        If `times` is None, a square matrix with all the offset versions of the
        first argument. Otherwise, `times` rows of a circulant matrix.
    """
    row = jnp.asarray(row) # Convert to JAX array 

    def scan_arg(carry, _): 
        new_carry = jnp.roll(carry, -1) 
        return (new_carry, new_carry)

    if times is None:
        nruter = jax.lax.scan(scan_arg, row, row)[1][::-1, :] 
    else:
        nruter = jax.lax.scan(scan_arg, row, None, length=times)[1][::-1, :]

    return nruter

def circulant_array(row, times=None):
    """Build a (full or partial) circulant matrix based on an array.

    Args:
        row: The first row of the matrix.
        times: If not None, the number of rows to generate.

    Returns:
        A circulant matrix based on the first row.
    """
    row = jnp.asarray(row)  # Convert to JAX array

    def scan_arg(carry, _):
        new_carry = jnp.roll(carry, -1)
        return new_carry, new_carry

    # If 'times' is None, generate a square circulant matrix
    if times is None:
        result = jax.lax.scan(scan_arg, row, None, length=row.shape[0])[1][::-1, :]
    else:
        result = jax.lax.scan(scan_arg, row, None, length=times)[1][::-1, :]

    return result

def positional_encoding(position: int, d_model: int) -> Array:
    """
    Generates a positional encoding matrix for input sequences using sine and cosine functions.

    Mathematical formulation:
        For each position pos and dimension i:
        PE(pos, 2i)   = sin(pos / (10000^(2i / d_model)))
        PE(pos, 2i+1) = cos(pos / (10000^(2i / d_model)))

    Args:
        position (int): The maximum length of the input sequence.
        d_model (int): The dimensionality of the embeddings.

    Returns:
        Array: A [position, d_model] 
    """

    # Create an array of positions (shape: [position, 1])
    pos = jnp.arange(position)[:, jnp.newaxis]

    # Create an array of dimensions (shape: [1, d_model])
    i = jnp.arange(d_model)[jnp.newaxis, :]

    # Compute the angle rates using the formula:
    # angle_rates = 1 / (10000^(2i / d_model))
    angle_rates = 1 / jnp.power(10000, (2 * (i // 2)) / jnp.float64(d_model))

    # Compute the positional angles (shape: [position, d_model])
    angle_rads = pos * angle_rates

    # Apply sine to even indices (2i)
    sines = jnp.sin(angle_rads[:, 0::2])

    # Apply cosine to odd indices (2i+1)
    cosines = jnp.cos(angle_rads[:, 1::2])

    # Concatenate sines and cosines to get the positional encoding matrix
    pos_encoding = jnp.concatenate([sines, cosines], axis=-1)

    return pos_encoding.astype(jnp.float64)

class BestIterKeeper:
    """Store the values of a bunch of quantities from the best iteration.

    "Best" is defined in the sense of lowest energy.

    Args:
        Hamiltonian: An array containing the Hamiltonian matrix.
        N: The number of spins in the chain.
        baseline: A lower bound for the V score. If the V score of the best
            iteration falls under this threshold, the process will be stopped
            early.
        filename: Either None or a file to write the best state to.
    """

    def __init__(
        self,
        Hamiltonian: npt.ArrayLike,
        N: int,
        baseline: float,
        filename: Optional[pathlib.Path] = None,
    ):
        self.Hamiltonian = Hamiltonian
        self.N = N
        self.baseline = baseline
        self.filename = filename
        self.vscore = np.inf
        self.best_energy = np.inf
        self.best_state = None

    def update(self, step, log_data, driver):
        """Update the stored quantities if necessary.

        This function is intended to act as a callback for NetKet. Please refer
        to its API documentation for a detailed explanation.
        """
        vstate = driver.state
        energystep = np.real(vstate.expect(self.Hamiltonian).mean)
        var = np.real(getattr(log_data[driver._loss_name], "variance"))
        mean = np.real(getattr(log_data[driver._loss_name], "mean"))
        varstep = self.N * var / mean**2

        if self.best_energy > energystep:
            self.best_energy = energystep
            self.best_state = copy.copy(driver.state)
            self.best_state.parameters = flax.core.copy(
                driver.state.parameters
            )
            self.vscore = varstep

            if self.filename != None:
                with open(self.filename, "wb") as file:
                    file.write(flax.serialization.to_bytes(driver.state))

        return self.vscore > self.baseline


@dataclass
class InvertMagnetization(MetropolisRule):
    """Monte Carlo mutation rule that inverts all the spins.

    Please refer to the NetKet API documentation for a detailed explanation of
    the MetropolisRule interface.
    """

    def transition(rule, sampler, machine, parameters, state, key, σ):
        indxs = jax.random.randint(
            key, shape=(1,), minval=0, maxval=sampler.n_chains
        )
        σp = σ.at[indxs, :].multiply(-1)
        return σp, None



def states_to_local_indices(inputs: Array, hilbert):
    return hilbert.states_to_local_indices(inputs)


def log_coeffs_to_log_psi(logCoeffs: Array, size: int, local_size: int,
                          positive_or_complex: str, softmaxOnOrOff: str,
                          modulusOnOrOff: str, signOfProbsIntoPhase: str):
    """
    Converts log coefficients from an autoregressive model into log-psi representation.
    
    Parameters:
    - logCoeffs: (size, local_size) array of coefficients.
    - size: Number of Hilbert space states.
    - local_size: Local degrees of freedom per site.
    - positive_or_complex: "positive" for real wavefunctions, "complex" for complex ones.
    - softmaxOnOrOff: "on" to apply softmax normalization, "off" for alternative methods.
    - modulusOnOrOff: "on" to normalize using modulus squared, "off" for raw values.
    - signOfProbsIntoPhase: "on" incorporates sign into phase, "off" does not.
    
    Returns:
    - log_psi: The log-wavefunction representation.
    """
    
    # Initialize phase and amplitude
    if softmaxOnOrOff == 'on':
        # Use softmax normalization
        phase = 1j * jnp.concatenate([jnp.zeros((size, 1)), logCoeffs[:, local_size - 1:]], axis=-1)
        amp = jnp.concatenate([jnp.zeros((size, 1)), logCoeffs[:, :local_size - 1]], axis=-1)
        if positive_or_complex == 'complex':
            log_psi = 0.5 * jax.nn.log_softmax(amp) + phase
        elif positive_or_complex == 'positive':
            log_psi = 0.5 * jax.nn.log_softmax(amp)
    
    elif softmaxOnOrOff == 'off':
        phase= 1j * jnp.concatenate([jnp.zeros((size, 1)), logCoeffs[:, local_size:]], axis=-1)
        amp= jnp.array(logCoeffs[:, :local_size]) # element 2
        if modulusOnOrOff == 'on':
            # Normalize using modulus squared (|amp|^2 / sum(|amp|^2))
            amp_abs_squared = jnp.abs(amp) ** 2
            cond_probs = amp_abs_squared / jnp.sum(amp_abs_squared, axis=-1, keepdims=True)
            log_psi = 0.5 * jnp.log(cond_probs) + phase if positive_or_complex == 'complex' else 0.5 * jnp.log(cond_probs)
        
        elif modulusOnOrOff == 'off':

            if signOfProbsIntoPhase == 'off':
                # Use absolute value of logCoeffs
                cond_probs = jnp.abs(amp) # activation function log_cosh(make it alwys positive)
                log_psi = 0.5 * jnp.log(cond_probs) + phase if positive_or_complex == 'complex' else 0.5 * jnp.log(cond_probs)

            elif signOfProbsIntoPhase == 'on' and positive_or_complex == 'complex':
                # Incorporate sign into phase
                phase = phase.at[:, 0].set(phase[:, 0] + 1j * jnp.pi / 4 * (1 - jnp.sign(amp[:, 0])))
                phase = phase.at[:, 1].set(phase[:, 1] + 1j * jnp.pi / 4 * (1 - jnp.sign(amp[:, 1])))
                cond_probs = jnp.abs(amp)
                log_psi = 0.5 * jnp.log(cond_probs) + phase

    return log_psi
