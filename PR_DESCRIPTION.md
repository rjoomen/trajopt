# Fix per-row coefficient handling in merit function evaluation

## Problem

Trajopt Ifopt fails to converge on trajectories with mixed Cartesian + collision constraints (e.g., constrained Cartesian endpoint with collision avoidance), while Trajopt SCO handles the same problems without issue. Both implement the same trust-region SQP algorithm from Schulman et al. 2013.

### Root Cause: Coefficient–Merit Function Mismatch

The bug is a consistency gap between how the QP subproblem weighs constraints and how the merit function evaluates constraint violations during step acceptance.

**In SCO** (`modeling_utils.cpp:247-268`), per-row coefficients are baked into the affine expression *before* it enters the QP:
```cpp
AffExpr aff = affFromValGrad(y[i], x_eigen, jac.row(i), vars_);
exprScale(aff, coeffs_[i]);   // coefficient is INSIDE the expression
```
Then `addAbs(aff, merit_coeff)` uses a **flat** merit coefficient for the slack penalty. The violation evaluation (`Constraint::value()`) also includes coefficients (`err *= coeffs_`), so the merit function sees `μ · Σ|coeffs[k] · c_k(x)|`.

**In master Ifopt** (`trajopt_qp_problem.cpp`), the coefficient placement is inverted:
```cpp
// convexify(): raw Jacobian, raw constant — no coefficient scaling
cc = cnt->getValues() - jac * x_initial;
// coefficient pushed into slack gradient instead
cache_slack_gradient.emplace_back(merit_coeff * info.coeffs(k));
```
The violation evaluation ignores coefficients entirely:
```cpp
calcBoundsViolations(err, val, bounds);
violations = err.sum();   // = Σ|c_k(x)|, NO per-row weighting
```

### Consequence

The QP correctly minimises `μ · Σ coeffs[k] · slack_k`, but the merit function evaluates `μ · Σ|c_k(x)|` without per-row weighting. Step acceptance decisions are therefore based on a different metric than what the QP actually optimised. This causes:

- **Incorrect step rejection**: Good QP steps that improve weighted violations get rejected because the unweighted merit doesn't reflect the improvement.
- **Unnecessary trust-region shrinkage**: Rejected steps trigger trust-region contraction, making subsequent steps more conservative.
- **Wrong penalty escalation**: Penalty coefficients inflate on the wrong constraints because the merit signal is distorted.
- **Convergence failure**: When Cartesian constraints (high coefficient weight) compete with collision constraints (many rows, unit weight), the unweighted merit is dominated by collision terms, drowning out Cartesian progress.

### Additional Bug: Convex cost evaluation used slack variables

`evaluateConvexCosts()` evaluated penalty costs using **all** QP variables (NLP + slack). Since the slack satisfies the equality `J·x + s⁺ − s⁻ = rhs` exactly at the QP solution, including slack columns makes the violation evaluate to ≈0. This caused the convex (approximate) merit to systematically underestimate penalty contributions, further distorting step acceptance.

---

## Changes

All changes are in `trajopt_qp_problem.cpp` (64 insertions, 27 deletions).

### 1. Pre-scale Jacobian, constant, and bounds by per-row coefficients in `convexify()`

**What:** Before building the QP constraint rows, scale each Jacobian row, the affine constant, and the constraint bounds by `info.coeffs(k)`. The slack gradient now uses a flat `merit_coeff` (the coefficient is already in the Jacobian).

**Why:** This matches SCO's `exprScale(aff, coeffs_[i])` followed by `addAbs(aff, merit_coeff)`. The QP constraint row becomes `coeffs[k]·J_k·x ∈ [coeffs[k]·lb − coeffs[k]·a_k, coeffs[k]·ub − coeffs[k]·a_k]` with slack penalty `μ·s`, which is mathematically equivalent to SCO's formulation.

**Zero-coefficient handling:** Rows with `coeffs[k] == 0` now get free bounds `[-∞, +∞]` and skip slack variable creation, matching SCO's `if (coeffs_[i] == 0) continue`. This also avoids OSQP degeneracy from zero-gradient slack columns in the KKT system.

### 2. Coefficient-weighted violation evaluation in all four evaluation methods

**What:**
- `evaluateConvexCosts()` and `evaluateConvexConstraintViolations()`: compare the pre-scaled linearised value against coefficient-scaled bounds (`coeffs[k] · bound`).
- `getExactCosts()` and `getExactConstraintViolations()`: weight raw violations element-wise by coefficients: `(err.array() * coefficients.array()).sum()`.

**Why:** The merit function must evaluate `μ · Σ|coeffs[k] · c_k(x)|` to match what the QP actually optimised. Without this, step acceptance is based on a different metric than the QP objective.

**SCO reference:**
- Exact: `Constraint::value()` bakes in coefficients (`err *= coeffs_`), then `violations()` takes `fabs()`/`pospart()` → result: `Σ|coeffs[k] · c_k(x)|`.
- Convex: `ConvexConstraints::violations()` evaluates `fabs(aff.value(x))` where `aff` already has the coefficient from `exprScale` → result: `|coeffs[k] · linearised_c_k(x)|`.

### 3. NLP-only variable selection in `evaluateConvexCosts()`

**What:** Restrict the Jacobian to `.leftCols(n_nlp_vars)` and multiply by `var_block` (NLP variables only) instead of the full variable vector including slack.

**Why:** SCO's `ConvexConstraints::violations()` evaluates `aff.value(x)` where `aff` only references NLP variables. Including slack columns makes the evaluated violation ≈0 (the QP equality is satisfied by construction), causing the convex merit to underestimate the actual penalty contribution.

---

## Verification

Each change has been mathematically verified against the SCO reference implementation:

| Aspect | SCO Reference | This PR | Match |
|--------|--------------|---------|-------|
| Jacobian scaling | `exprScale(aff, coeffs_[i])` | `scaled_val = val * info.coeffs(row)` | ✓ |
| Constant scaling | Same `exprScale` | `cc.array() *= info.coeffs.array()` | ✓ |
| Bounds scaling | Implicit (aff includes coeff) | `bound * info.coeffs(k) - constant` | ✓ |
| Flat slack gradient | `addAbs(aff, merit_coeff)` | `cache_slack_gradient(merit_coeff)` | ✓ |
| Zero-coeff skip | `if (coeffs_[i] == 0) continue` | Free bounds + no slack | ✓ |
| Exact violation weighting | `value()` bakes in coeffs | `(err * coeffs).sum()` | ✓ |
| Convex violation weighting | `fabs(aff.value(x))` with scaled aff | `max(0, val-c*ub) + max(0, c*lb-val)` | ✓ |
| Convex cost: NLP vars only | `aff` uses only NLP vars | `.leftCols(n_nlp_vars)` | ✓ |

## Test plan

- [ ] Verify existing unit tests pass
- [ ] Test trajectory optimisation with Cartesian endpoint constraint + collision avoidance (the scenario that previously failed to converge)
- [ ] Compare convergence behaviour (iteration count, merit trace) against SCO on the same problem
- [ ] Verify that problems without per-row coefficients (all coefficients = 1) produce identical results to master
