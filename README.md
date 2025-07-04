# RandomATF_Entanglement_VMC
Code for calculating properties (entanglement and otherwise) of random ATF wavefunctions, as well as to perform VMC.

# 🧠 Random Autogressive Transformer (ATF) Wavefunctions for Quantum Spin Systems

This part implements an ATF to model the wavefunction of a quantum spin chain. It supports a variety of hyperparameters including attention modes, activation functions, and sampling strategies. The model allows evaluation of Renyi entanglement entropy, spin-spin correlation functions, and the Schmidt spectrum (eigenvalues of reduced density matrices).

---

## How to Run calculations of static properties for random ATF states

Use the following `srun` command to execute the script on a single GPU node:

```bash
srun --ntasks=1 --exclusive --gpus=1 --cpus-per-gpu=5 --mem=50G \
python spin_S_An.py \
$N $J $h $num_layers $num_heads $dff $d_model $embedding_var \
$positive_or_complex $softmaxOnOrOff $modulusOnOrOff $signOfProbsIntoPhase \
$ARD_vs_MCMC $run $attn_mode $activation_fn $Quantity >> Auto_results/try/output.log 2>&1 &
```

### Argument Descriptions

J=-1
h=-1
num_layers=1
dff=20
num_heads=2
mkdir -p Auto_results/try
positive_or_complex="complex"
softmaxOnOrOff="on"
modulusOnOrOff="off"
signOfProbsIntoPhase="off"
ARD_vs_MCMC="ARD"
attn_mode="s"
Quantity="Cor"
activation_fn="relu"

| Argument               | Example                     |Description                                                                                               
| `N`                    | `20`                        | Number of spins (sites) in the chain                                                                                  |
| `J`                    | `-1`                        | Ising interaction strength (negative = antiferromagnetic)                                                             |
| `h`                    | `-1`                        | External magnetic field strength                                                                                      |
| `num_layers`           | `1`                         | Number of Transformer layers                                                                                          |
| `num_heads`            | `2`                         | Number of attention heads                                                                                             |
| `dff`                  | `20`                        | Feed-forward layer dimension                                                                                          |
| `d_model`              | `40`                        | Transformer embedding size                                                                                            |
| `embedding_var`        | `0.1`                       | Gaussian distribution width                                                                                           |
| `positive_or_complex`  | `"complex"` or `"positive"` | Wavefunction representation type  (complex or real)                                                                   |
| `softmaxOnOrOff`       | `"on"` or `"off"`           | Whether to apply softmax at output layer (g=SM)                                                                       |
| `modulusOnOrOff`       | `"on"` or `"off"`           | Whether using a square modulus function   (g=MOD)                                                                     |
| `signOfProbsIntoPhase` | `"on"` or `"off"`           | Whether to map sign of probabilities into phase                                                                       |
| `ARD_vs_MCMC`          | `"ARD"` or `"MCMC"`         | Sampling method: ARDirect or MetropolisLocal                                                                          |
| `run`                  | `0, 1, 2...`                | Run index (controls RNG seed)                                                                                         |
| `attn_mode`            | `"s"` or `"c"`              | Attention `s`: softmax ,`c`: circulant attention                                                                      |
| `activation_fn`        | `"relu"` or `"I"`           | Activation function:`relu`: ReLU , `I`: identity function                                                             |
| `Quantity`             | `"Ent"`, `"Cor"`, `"Sch"`   | Physical quantity to compute:`Ent`: Renyi-2 Entropy, `Cor`: Spin-Spin Correlation, `Sch`: Schmidt eigenvalues         |



## How to Run VMC optimizations on our ATF wavefunctions

Use the following `srun` command to execute the script on a 4 GPUs node:

```bash
srun --ntasks=1 --exclusive --gpus=4 --cpus-per-gpu=5 --mem=50G python vmc_ATF.py $N $J $h $num_layers $num_heads $dff "$d_model_formatted" "$embedding_var_formatted" $positive_or_complex $softmaxOnOrOff $modulusOnOrOff $signOfProbsIntoPhase $attn_mode $activation_fn $seed 

```

### Argument Descriptions

J=-1
h=-1
num_layers=1
dff=20
num_heads=2
positive_or_complex="complex"
softmaxOnOrOff="on"
modulusOnOrOff="off"
signOfProbsIntoPhase="off"
attn_mode="s"
activation_fn="relu"
