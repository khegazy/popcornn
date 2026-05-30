# Concepts

A one-page background for the physics popcornn is solving and why it's
set up the way it is.

## Reactions, energy surfaces, and barriers

A molecule's energy depends on where its atoms are. If you plot energy
as a function of all atomic coordinates, you get a **potential energy
surface (PES)**. A stable molecule sits in a local minimum of the PES.

A chemical reaction takes the system from one minimum (the
**reactant**) to another (the **product**). The minima are connected
by paths through the surface, and any such path has to climb at least
once before it can come down — there's an energy hill in the way.

The path that minimizes the height of that hill, going through the
**lowest** climb-then-descend, is called the **minimum-energy path
(MEP)**. The single highest point on the MEP is called the
**transition state (TS)**. The energy difference between the
reactant minimum and the TS is the **barrier height** — usually the
quantity you actually care about, because it controls how fast the
reaction goes.

So finding the reaction means finding the MEP, and finding the
barrier means finding the TS.

## Why this is hard

A molecule with $N$ atoms has $3N$ coordinates, so the PES lives in
$3N$ dimensions. You can't visualize it, you can't grid it, and the TS
is a *saddle point* — a maximum along one direction (the reaction
coordinate) and a minimum along all the others. Standard local
optimizers go to minima, not saddles, so finding the TS requires
specialized methods.

Two families of methods dominate:

- **String / NEB methods** — represent the path as a discrete chain of
  images and project forces along/perpendicular to the chain so the
  chain settles into the MEP. Cheap, well-understood, but the
  discretization is coarse and the chain can get stuck.
- **Surface walking and dimer methods** — start from a guess near the
  TS and follow the unstable mode upward. Fast but require a good
  initial guess.

Popcornn is a third option: represent the path as a small **neural
network** that maps a scalar time $t \in [0, 1]$ to a configuration
$x(t)$, with $x(0) = $ reactant and $x(1) = $ product, and train the
network so that the path crosses the lowest barrier the energy model
can find.

## Why use a neural network for the path?

Because once the path is parameterized as a function $x(t)$ with
trainable weights $\theta$, you can:

1. Evaluate the energy and its gradient at any number of points along
   the path, with no fixed discretization.
2. Define a **loss** that's an integral over $t \in [0, 1]$ of some
   per-point quantity (a force-velocity overlap, for example).
3. Backpropagate $\partial \text{loss} / \partial \theta$ through the
   path, the energy model, and the integrator, and step an optimizer.

That's the whole training loop. Adaptive quadrature
([torchpathint](https://github.com/khegazy/torchpathdiffeq))
chooses where along the path to evaluate, so dense regions get more
points and flat regions get fewer.

## What loss?

Different choices give different physical interpretations. The two
most-used in popcornn are:

- **Geodesic loss** — minimizes path length on a metric defined by the
  decomposed forces. With a repulsive potential this gives geodesic
  interpolation and resolves atom clashes. Good as a pre-step.
- **Projected variational reaction energy (pVRE)** —
  $\int_0^1 |\mathbf{v}(t) \cdot \mathbf{F}(t)| \, \mathrm{d}t$, where
  $\mathbf{v}$ is the path velocity and $\mathbf{F}$ is the force.
  Minimizing this drives the path toward configurations where the
  force is perpendicular to the path direction — the saddle-point
  condition.

There's a list of all available loss terms in
[Loss functions](loss-functions.md).

## What about the transition state?

After the path converges, popcornn scans along it for the highest
energy point and returns that as the predicted TS (`popcornn_ts.xyz`
in the YAML driver, the second return of `optimize_path` in the
Python API).

This is a **good guess**, not a guarantee. To get a publication-grade
TS:

1. Take the popcornn TS as a starting point.
2. Run a saddle-point optimizer (e.g.
   [Sella](https://github.com/zadorlab/sella)) to converge to the
   actual saddle.
3. Run intrinsic-reaction-coordinate calculations forward and reverse
   from the saddle to verify it connects your reactant and product.

Popcornn's job is to give you that good starting point cheaply.

## Where to go next

- [Loss functions](loss-functions.md) for the per-point quantities
  popcornn integrates.
- [Paths](paths.md) for the neural-network path representation
  (`width`, `depth`, what `MLPpath` is built on top of `LinearPath`).
- [Potentials](potentials.md) for which energy models are wired up.
