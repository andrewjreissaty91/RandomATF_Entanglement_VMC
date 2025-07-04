import argparse
import os
import sys
os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"
from transformer_LR_WF.autoregressive_An import *
from jax import tree_util
import netket as nk
import netket.experimental as nkx
import numpy as np
import optax
from netket.operator.spin import sigmax,sigmaz 
nk.config.netket_spin_ordering_warning = False  

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
    "attn_mode", type=str, help="attention_mode"
)
argument_parser.add_argument(
    "activation", type=str, help="activation function: ReLU or I"
)
argument_parser.add_argument(
    "seed", type=int, help="seed"
)
args = argument_parser.parse_args()

N = args.N
J = args.J
h = args.h
num_layers = args.num_layers
num_heads = args.num_heads
dff = args.dff  
d_model = int(args.d_model)
embedding_var = args.embedding_var
positive_or_complex = args.positive_or_complex
softmaxOnOrOff = args.softmaxOnOrOff
modulusOnOrOff = args.modulusOnOrOff
signOfProbsIntoPhase = args.signOfProbsIntoPhase
attn_mode= args.attn_mode
activation_name = args.activation
seed = args.seed

hi = nk.hilbert.Spin(s=0.5, N=N)
H = sum (J * sigmaz(hi, i) * sigmaz(hi, (i+1)%N) for i in range(N-1) ) + sum(h* sigmax(hi, i) for i in range(N)) 
H = H.to_jax_operator()
if activation_name.lower() == "relu":
    activation_fn = nn.relu
elif activation_name.lower() == "i":
    activation_fn = lambda x: x
else:
    raise ValueError(f"Unsupported activation function: {activation_name}")
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
base_lr = 1e-3
max_iters = 5000
sampler = nk.sampler.ARDirectSampler(hi)
optimizer = nk.optimizer.Adam(learning_rate=base_lr)
vstate = nk.vqs.MCState(sampler=sampler, model=model, n_samples=1024, seed=seed)
gs = nk.driver.VMC(H,optimizer,variational_state=vstate)
run = int(seed/100)
output_file = f"VMC/try/{d_model}/{embedding_var}/output_N{N}_demb{d_model}_var{embedding_var:.2f}_run{run}"
if not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))
gs.run(n_iter=max_iters, out=output_file)
