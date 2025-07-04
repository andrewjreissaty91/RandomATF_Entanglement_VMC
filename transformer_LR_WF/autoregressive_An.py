# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Tuple
import jax
import netket as nk
from flax import linen as nn
from jax import numpy as jnp
from functools import partial
from netket.nn.masked_linear import default_kernel_init
from netket.utils.types import NNInitFunc
from netket.hilbert.homogeneous import HomogeneousHilbert
from flax.linen.initializers import zeros
from netket.utils.struct import dataclass 
from netket.models import AbstractARNN
from .utils_An import positional_encoding, circulant_array, REAL_DTYPE,states_to_local_indices, log_coeffs_to_log_psi
PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any
DotGeneralT = Callable[..., Array]
ConvGeneralDilatedT = Callable[..., Array]


class AffinityPosweight(nn.Module):
    """circulant matrix""" 
    """for calculate the attention weights"""
    head_size: int
    embedding_var: float
    attn_mode: str
    use_hidden_bias: bool = False
    
    @nn.compact
    def __call__(self, x, mask):
        if self.attn_mode == "s": 
            query = nn.Dense(features=self.head_size, kernel_init=nn.initializers.normal(stddev=self.embedding_var), param_dtype=REAL_DTYPE, use_bias=self.use_hidden_bias)(x) # shape = (seq_len, head_size)
            value = nn.Dense(features=self.head_size, kernel_init=nn.initializers.normal(stddev=self.embedding_var), param_dtype=REAL_DTYPE, use_bias=self.use_hidden_bias)(x) # shape = (seq_len, head_size)
            key= nn.Dense(features=self.head_size, kernel_init=nn.initializers.normal(stddev=self.embedding_var), param_dtype=REAL_DTYPE, use_bias=self.use_hidden_bias)(x) # shape = (seq_len, head_size)
            weight = jnp.matmul(query, key.T) / jnp.sqrt(self.head_size)
            if mask is not None:
                weight = weight + (mask * -1e9)  
            weight = nn.softmax(weight, axis=-1)
            attention = jnp.matmul(weight, value)
        elif self.attn_mode == "c":
            value = nn.Dense(features=self.head_size, kernel_init=nn.initializers.normal(stddev=self.embedding_var), param_dtype=REAL_DTYPE, use_bias=self.use_hidden_bias)(x) # shape = (seq_len, head_size)
            weight_row = self.param(
            "alpha_delta",
            nn.initializers.truncated_normal(
                stddev=self.embedding_var,
            ),
            (x.shape[-2],),
            REAL_DTYPE,)
            weight =  circulant_array(weight_row)
            if mask is not None:
               weight  = weight  *  mask 
            attention=  weight @ value
        return attention

class PositionalHead(nn.Module):
    """Positional head module"""
    head_size: int
    """dimension of the head"""
    embedding_var: float
    attn_mode: str 
    use_hidden_bias: bool = False
    @nn.compact
    def __call__(self, x, mask):
        aff = AffinityPosweight(self.head_size, self.embedding_var, self.attn_mode)
        return aff(x, mask)

class MultiHeadAttention(nn.Module):
    """Multi-head attention module"""
    num_heads: int
    """number of attention heads"""
    head_size: int 
    """dimension of each head"""
    embedding_var: float
    attn_mode: str 
    @nn.compact
    def __call__(self, x, mask):
        heads = [PositionalHead(self.head_size, self.embedding_var, self.attn_mode) for _ in range(self.num_heads)]  
        return jnp.concatenate([h(x, mask) for h in heads], axis=-1)

    
class PointWiseFeedForwardNetwork(nn.Module):
    """Point-wise feed forward neural network"""
    d_model: int
    """hidden layer size"""
    dff: int
    """DFF"""
    embedding_var: float
    activation_fn: Callable = nn.relu
    """activation function"""
    use_hidden_bias: bool = False
    @nn.compact
    def __call__(self, x):
        x_1 = nn.Dense(features=self.dff,kernel_init=nn.initializers.normal(stddev=self.embedding_var), param_dtype=REAL_DTYPE, use_bias=self.use_hidden_bias)(x)
        h_1 = self.activation_fn(x_1)
        x_2 = nn.Dense(features=self.d_model,kernel_init=nn.initializers.normal(stddev=self.embedding_var), param_dtype=REAL_DTYPE, use_bias=self.use_hidden_bias)(h_1)
        return x_2

class DecoderLayer(nn.Module):
    d_model: int
    """hidden layer size"""
    dff: int
    """DFF"""
    num_heads: int
    """number of attention heads"""
    embedding_var: float
    attn_mode: str
    activation_fn: Callable = nn.relu


    @nn.compact
    def __call__(self, x, look_ahead_mask):
        head_size = self.d_model // self.num_heads
        mha = MultiHeadAttention(self.num_heads, head_size, self.embedding_var,self.attn_mode)
        x+= mha(x, look_ahead_mask) 
        x = nn.LayerNorm(epsilon=1e-6, param_dtype=REAL_DTYPE)(x)
        ffn = PointWiseFeedForwardNetwork(self.d_model,self.dff,self.embedding_var,self.activation_fn)
        x+= ffn(x)
        x = nn.LayerNorm(epsilon=1e-6)(x)
        return x



class Decoder(nn.Module):
    """Decoder consisting of the positional embedding and Multi-head attention layers"""
    hilbert: HomogeneousHilbert
    """the Hilbert space. Only homogeneous unconstrained Hilbert spaces are supported."""
    num_layers: int
    """number of attention layers"""
    d_model: int
    """hidden layer size"""
    dff: int
    """DFF"""
    num_heads: int

    embedding_var: float
    attn_mode: str 
    activation_fn: Callable = nn.relu
    """number of attention heads
    Args:  
        x: shape (Hilbert.size, )
        look_ahead_mask: shape (Hilbert.size, Hilbert.size)
    Returns:
        x: shape (Hilbert.size, d_model)
    """

    @nn.compact
    def __call__(self, x, look_ahead_mask):
        seq_len = x.shape[0] 
        embedding = nn.Dense(features=self.d_model, kernel_init=nn.initializers.normal(stddev=self.embedding_var), param_dtype=REAL_DTYPE)
        x = embedding(jax.nn.one_hot(x, self.d_model))
        x *= jnp.sqrt(self.d_model) 
        pos_encoding = positional_encoding(self.hilbert.size, self.d_model)
        x += pos_encoding[:seq_len, :] 
        # Decoder layers  
        dec_layers = [DecoderLayer(d_model=self.d_model,
                                        num_heads=self.num_heads,
                                        dff=self.dff, embedding_var=self.embedding_var, attn_mode=self.attn_mode,activation_fn=self.activation_fn) for _ in range(self.num_layers)]

        for i in range(self.num_layers):
            x = dec_layers[i](x, look_ahead_mask) 

        return x

class Transformer(AbstractARNN):
    """Transformer Wavefunction (Adapted from Juan Carrasquilla's Tensorflow implementation)"""
    hilbert: HomogeneousHilbert
    """the Hilbert space. Only homogeneous unconstrained Hilbert spaces are supported."""
    autoreg: bool
    """Whether the model is autoregressive or not"""
    num_layers: int
    """number of attention layers"""
    d_model: int
    """hidden layer size"""
    dff: int
    """DFF"""
    num_heads: int
    """number of attention heads"""
    embedding_var: float
    positive_or_complex: str
    softmaxOnOrOff: str
    modulusOnOrOff: str
    signOfProbsIntoPhase: str
    attn_mode: str 
    activation_fn: Callable = nn.relu 
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weights."""
    bias_init: NNInitFunc = zeros
    """initializer for the biases."""

    def setup(self):
        self.L = self.hilbert.size
        self.decoder = Decoder(hilbert=self.hilbert,
                               num_layers=self.num_layers,
                               d_model=self.d_model,
                               dff=self.dff,
                               num_heads=self.num_heads,
                               embedding_var=self.embedding_var,
                               attn_mode=self.attn_mode, activation_fn=self.activation_fn

                               )

        if self.softmaxOnOrOff == 'on':
            self.outputdense = nn.Dense(features=(self.hilbert.local_size - 1) * 2, # element 2
                                kernel_init=nn.initializers.normal(stddev=self.embedding_var),
                                bias_init=self.bias_init,
                                param_dtype=REAL_DTYPE)
        elif self.softmaxOnOrOff == 'off':
            self.outputdense = nn.Dense(features=self.hilbert.local_size + 1, # element 3
                                    kernel_init=nn.initializers.normal(stddev=self.embedding_var),
                                    bias_init=self.bias_init,
                                    param_dtype=REAL_DTYPE)

        if self.autoreg:
            if self.attn_mode == "s":
                self.mask = 1 - jnp.tril(jnp.ones((self.hilbert.size, self.hilbert.size)))
            elif self.attn_mode == "c":
               
                self.mask = jnp.tril(jnp.ones((self.hilbert.size, self.hilbert.size)))
            else:
                raise ValueError(f"Unknown attn_mode: {self.attn_mode}")
        else:
            self.mask = None


    def conditionals(self, inputs: Array) -> Array:
        """
        Computes the conditional probabilities for each site to take each value.

        Args:
          inputs: configurations with dimensions (batch, Hilbert.size).

        Returns:
          The probabilities with dimensions (batch, Hilbert.size, Hilbert.local_size).

        """
        inputs_dim = inputs.ndim
        # if there is only one batch dimension, expand with a one
        if inputs_dim == 2:
            inputs = jnp.expand_dims(inputs, axis=0)
        inputs = jnp.reshape(inputs, (-1, *inputs.shape[2:]))
        #convert (+1, -1, +1, -1) to (0, 1, 0, 1)
        inputs = states_to_local_indices(inputs, self.hilbert)
        log_psi = conditionals_log_psi(inputs,
                                       self.mask,
                                       self.hilbert.local_size,
                                       self.hilbert.size,
                                       self.decoder,
                                       self.outputdense,
                                       self.positive_or_complex,
                                       self.softmaxOnOrOff,
                                       self.modulusOnOrOff,
                                       self.signOfProbsIntoPhase)
        p = jnp.exp( 2* log_psi.real)
        return p 

    @nn.compact

    def __call__(self, inputs: Array) -> Array:

        inputs_dim = inputs.ndim 
        if inputs_dim == 2: 
            inputs = jnp.expand_dims(inputs, axis=0)
        # ----reshape inputs --(1, Hilbert.size) ----
        batch_shape = list(inputs.shape[:2]) 
        inputs = jnp.reshape(inputs, (-1,inputs.shape[-1])) 
        inputs = states_to_local_indices(inputs, self.hilbert)
        # --do transformer ----
        log_psi = conditionals_log_psi(inputs,
                                       self.mask,
                                       self.hilbert.local_size,
                                       self.hilbert.size,
                                       self.decoder,
                                       self.outputdense,
                                       self.positive_or_complex,
                                       self.softmaxOnOrOff,
                                       self.modulusOnOrOff,
                                       self.signOfProbsIntoPhase)
        # ---- log_psi is now of shape (Hilbert.size, local_state) ----
        one_hot_samples = nn.one_hot(inputs, self.hilbert.local_size, axis=-1)
        log_psi = (log_psi * one_hot_samples).sum(axis=(1, 2))
        # ---- log_psi is now of shape (batch*Hilbert.size) ----
        if inputs_dim == 2:
            return log_psi
        else:
            return jnp.reshape(log_psi, batch_shape)



@partial(jax.vmap, in_axes=(0, None, None, None, None, None, None, None, None, None))
def conditionals_log_psi(x: Array, mask: Array, local_size: int, size: int, decoder: Callable,
                         outputdense: Callable, positive_or_complex: str, softmaxOnOrOff: str, modulusOnOrOff: str, signOfProbsIntoPhase: str) -> Array:
    """
    Computes the logarithmic wave function for each site to take each value.

    Args:
      x: configurations with dimensions (Hilbert.size).

    Returns:
      The logarithmic wavefunction with dimensions (Hilbert.size, Hilbert.local_size).

    """
    # input shape(1, Hilbert.size)
    init = jnp.zeros(1, dtype=jnp.int32)
    # output shape (Hilbert.size + 1)
    output = jnp.concatenate([init, x], axis=0)
    output = output[0:size]
    # ---- output is now of shape (Hilbert.size) ----
    dec_output = decoder(output, mask)
    # ---- dec_output is now of shape (Hilbert.size, d_model) ----
    output_ampl = outputdense(dec_output) 
    # ---- output_ampl is now of shape (Hilbert.size, (Hilbert.local_size - 1) * 2) ----
    log_psi = log_coeffs_to_log_psi(output_ampl + 1e-14,
                                    size=size,
                                    local_size=local_size,positive_or_complex=positive_or_complex,softmaxOnOrOff=softmaxOnOrOff,modulusOnOrOff=modulusOnOrOff,signOfProbsIntoPhase=signOfProbsIntoPhase)
   
    # ---- log_psi is now of shape (Hilbert.size, Hilbert.local_size) ---- 
 
    return log_psi



