# ExSpectra

This anonymous release contains the importable method components for
**ExSpectra**, a co-adaptive replay valuation rule for cooperative multi-agent
reinforcement learning. It follows the paper and appendix definitions while
leaving the learner, environment, evaluation policy, and data store to the
host project.

The release is intentionally limited to the method. It does not contain
baseline implementations, benchmark environments, experiment logs, model
weights, private traces, manuscript sources, or numerical experiment settings.
No standalone training script or command-line entry point is included. The
Python files are library modules and are meant to be imported by an external
experiment implementation.

## Method contract

At checkpoint `X_k`, the host learner supplies a candidate pool `E_k`, a
candidate-independent reference batch `B_k^0`, and a fixed continuation
protocol `P_k`. The initial focal-only SGD displacement for candidate `e` in
slot `iota` is

```text
d[e, iota] = -(eta_0 / B) * (grad_R loss(e) - grad_R loss(e_iota^0))
```

The three matched branches obey the appendix protocol:

* `frz`: teammate-owned state is held fixed;
* `scr`: teammate state follows the candidate-independent co-adaptive
  reference path and is copied at continuation boundaries;
* `co`: teammates continue learning normally.

Their exact CAUD identity is

```text
u_co(e) = u_frz(e) + [u_scr(e) - u_frz(e)] + [u_co(e) - u_scr(e)]
```

The bracketed terms are respectively `Gamma_comp(e)` (reference-learning
complementarity) and `Gamma_resp(e)` (candidate-induced response).

For the fixed-target local approximation, the host autodiff implementation
provides the contracted products from the focal and teammate parameter blocks.
The package assembles

```text
b_frz  = g_R - h_own - C_RR
b_comp = -C_RT
b_resp = -h_resp
u_tilde(e) = dot(d_e, b_frz + b_comp + b_resp)
```

The reference-batch gradient is shared across candidates. For non-shared
policy blocks, the cross-agent product is evaluated from fixed trajectories as
`mean(R * focal_score * dot(teammate_score, v_T))`, avoiding a dense
cross-agent Hessian.

## Calibration and allocation

Completed probes are stored with their source checkpoint, frozen structural
prediction, pre-probe features, source propensity, and target propensity. A
`RidgeCalibrator` fits only records whose source checkpoint is earlier than the
current decision checkpoint, using the appendix importance weight
`w_a = nu_a(e_a) / mu_a(e_a)`.

The allocator samples exactly the caller-supplied replay budget without
replacement. At every draw it mixes a score softmax and a uniform component;
the temperature and mixture coefficient are explicit function arguments rather
than embedded experiment settings.

## Module map

* `displacement` — focal SGD displacement and reference anchor;
* `influence` — score-function estimates and shared CAUD vectors;
* `caud` — exact branch decomposition and structural score assembly;
* `audit` — positive-support probe draws and matched branch labels;
* `calibration` — temporally valid weighted ridge residual fitting;
* `allocation` — sequential score/uniform mixture sampling.

The modules accept arrays or tensors supplied by the host learner. They do not
choose a QMIX architecture, optimizer, horizon, evaluation temperature, probe
count, replay budget, or any other experimental value. Those values must remain
in the private experiment configuration.

## Example integration shape

An external learner can import the components and provide its own gradients,
autodiff products, branch runner, and evaluation function:

```python
from influence import build_shared_coefficients
from displacement import candidate_displacement
from allocation import sequential_mixture_sample
from caud import structural_score

coefficients = build_shared_coefficients(
    g_focal, curvature_rr, curvature_rt, sensitivity_own, sensitivity_response
)
displacements = [
    candidate_displacement(
        gradient, reference_gradient,
        step_size=caller_step,
        batch_size=caller_batch_size,
    )
    for gradient in candidate_gradients
]
raw_scores = structural_score(displacements, coefficients)
selected = sequential_mixture_sample(
    candidates,
    raw_scores,
    batch_size=caller_replay_budget,
    temperature=caller_temperature,
    uniform_mix=caller_uniform_mixture,
)
```

The snippet documents the interface only; this release does not provide a
directly executable training program.

## Scope and audit discipline

The host project must keep the complete learner checkpoint, optimizer/target
state, masks, fixed continuation data, random keys, branch marginals, and
resource accounting. Current-checkpoint audit outcomes must be appended only
after scoring and become eligible for later calibration decisions. The
anonymous mirror is intended for implementation inspection and method-level
integration; it is not a complete reproduction bundle.
