# popcornn

**Pop**cornn — *Path Optimization with a Continuous Representation Neural
Network* — is a tool for finding the path a chemical reaction takes
between a known reactant and a known product.

You give popcornn the two endpoint structures and an energy model. It
fits a smooth, differentiable path between them and adjusts that path so
that it crosses the lowest-energy barrier the energy model can find.

The path is represented by a small neural network, which means the whole
thing optimizes by gradient descent in PyTorch — no force-evaluations on
fixed grids, no string-method bookkeeping.

## What popcornn is good for

- Finding **transition states** when you have a guess for the reactant
  and product but not the saddle point in between.
- **Reaction-path interpolation** that's better than a straight line —
  popcornn fits the path through any intermediate frames you provide
  and bends it around atom clashes using a repulsive pre-step.
- Working with **machine-learned interatomic potentials** (MACE, Orb,
  UMA, …) on systems too large for DFT-driven NEB.

## What popcornn is *not*

Popcornn targets the transition state directly, so the converged path
is a **good guess** at the minimum-energy path, not a guaranteed
minimum-energy path. For publication-grade results, follow up with a
saddle-point optimization (e.g. [Sella](https://github.com/zadorlab/sella))
and intrinsic reaction coordinate calculations on the popcornn
transition state.

## How to read these docs

If you've never used popcornn before:

1. [Getting Started](getting-started.md) — install and run your first
   reaction.
2. [Concepts](concepts.md) — what's a reaction path, transition state,
   MEP. Skip if you already know.
3. [Configuration reference](configuration.md) — every YAML key.

When you hit a wall:

- [Convergence](convergence.md) — picking `threshold` and `patience`.
- [Memory & OOM](memory-and-oom.md) — what to do when CUDA runs out.

When you want more:

- [Potentials](potentials.md), [Paths](paths.md),
  [Loss functions](loss-functions.md) — what's built in, how to add
  your own.
- [Advanced](advanced.md) — schedulers, multi-stage optimization,
  transition-state losses.
- [Derivation](derivation.md) — the math behind the closed-form
  recipe `threshold = δ · 2·σ_min · F2_target`.
- [Code reference](reference/popcornn.md) — auto-generated API docs.
