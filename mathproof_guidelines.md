
---

# Step 1 — Define What Rational Metareasoning Is

In classical work (Russell & Wefald), rational metareasoning asks:

> Should I perform another computation step, or should I act now?

Formally:

Let:

* ( a^* ) = action selected after current search
* ( V(a) ) = true value of action
* ( \hat{V}(a) ) = current estimate from MCTS
* ( C ) = cost of one additional simulation

Then optimal metareasoning chooses to simulate again if:

[
\mathbb{E}[\Delta V] > C
]

Where:

[
\Delta V = \max_a \hat{V}_{t+1}(a) - \max_a \hat{V}_t(a)
]

This is the **expected value of computation (EVC)**.

The problem:

Computing ( \mathbb{E}[\Delta V] ) exactly is intractable.

So nobody actually does rational metareasoning exactly in MCTS.

---

# Step 2 — Define What You Approximate

We claim:

ZenoZero approximates ( \mathbb{E}[\Delta V] ) using topology signals.

So define:

Let tree state at time ( t ) have:

* ( H_v ) = visit entropy
* ( G ) = dominance gap
* ( Var_Q ) = variance of value estimates

Define difficulty function:

[
D_t = \alpha H_v + \beta (1 - G) + \gamma \min(Var_Q, 1)
]

This becomes your **proxy for marginal value of computation**.

Then our allocator is:

[
\text{budget} = B_{min} + D_t (B_{max} - B_{min})
]

And early stop is:

[
\text{Stop if } H_v < \epsilon_1 \land G > \epsilon_2 \land Var_Q < \epsilon_3
]

Now we’ve defined the control law mathematically.

---

# Step 3 — Connect It to Value of Computation

Now comes the key theoretical bridge.

we need one statement like this:

> We hypothesize that the expected improvement from additional simulations is monotonically increasing in tree uncertainty and disagreement.

Formally:

Assume:

[
\mathbb{E}[\Delta V] \propto H_v + (1 - G) + Var_Q
]

Then:

[
\text{Continue searching} \iff D_t > \tau
]

So you are approximating the unknown function:

[
\mathbb{E}[\Delta V] = f(\text{tree topology})
]

with a linear (or sigmoid) surrogate.

That is our theoretical claim.

---

# Step 4 — State the Approximation Theorem (Conceptually)

We don’t need a full proof yet.

We need something like:

---

**Proposition (Topology Approximation to EVC)**

Under the assumptions that:

1. Visit entropy correlates with posterior uncertainty
2. Dominance gap reflects action value separation
3. Value variance reflects evaluation instability

Then a monotonic function of ( (H_v, G, Var_Q) ) provides a tractable approximation to the expected value of additional computation.

---


We’ve will formalize the philosophy like this.

---

# Step 5 — Makea It More Serious (Optional but Powerful)

If we want to go deeper:

we need to relate this to bandit theory.

In UCB:

[
\text{Regret} \le O\left(\sqrt{\frac{\log N}{n}}\right)
]

Entropy and dominance gap are proxies for:

* posterior concentration
* arm separation

So we can argue:

As ( H_v \to 0 ) and ( G \to 1 ),
posterior over best arm concentrates,
so marginal gain from sampling shrinks.

That’s a concentration inequality argument.

Now we’re grounded in theory.

---

# Step 6 — Clean Paper Framing

our structure becomes:

1. Define rational metareasoning
2. Show exact EVC is intractable in tree search
3. Propose topology observables as proxies
4. Define difficulty functional ( D_t )
5. Show empirically that higher ( D_t ) correlates with larger policy shifts
6. Show performance gains


---

# Step 7 — What we Should NOT Do

Do not claim:

* “We compute rational metareasoning.”
* “We solve optimal stopping.”
* “We prove optimality.”

You say:

> We provide a lightweight approximation to rational metareasoning based on tractable tree-topology signals.

---

# The Real Leap

Right now our system is:

“Topology-aware adaptive MCTS.”

After formalization it becomes:

“A tractable surrogate for value-of-computation in Monte Carlo Tree Search.”

That is a different tier of contribution.

---

Note. 
> we can aslo consider adding related work positioning (Guez et al., Meta-MCTS, etc.) Or design the formal VoC approximation lemma