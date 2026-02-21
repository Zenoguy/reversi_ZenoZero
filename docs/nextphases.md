# Phase 5.5 Through 10: Conceptual Roadmap

---

## Phase 5.5 — Learning Instead of Engineering

### **The Core Problem**

Right now, your λ formula uses three fixed weights: 30% for tree concentration (1-H_v), 30% for dominance (G), and 40% for value stability (1-Var_Q). These numbers came from manual experimentation on Reversi. But think about what this means:

**The metareasoning policy is a hand-crafted heuristic.** You've built a system that decides when to trust heuristics... using a heuristic. That's philosophically awkward and practically limiting.

When you move to Chess, those weights will be wrong. Chess has deeper tactics, longer horizons, more complex positions. The balance between "tree looks concentrated" and "values are stable" will be different. You'll have to retune. Same for Go, same for StarCraft, same for robotics.

### **The Conceptual Shift**

Instead of engineering the metareasoning policy, **learn it from experience**. The system should discover, through trial and error, what topology patterns predict when heuristics are trustworthy.

Think of it like this: Right now you're telling the system "when entropy drops below X and dominance exceeds Y, trust your heuristic." That's prescriptive. The learned version says "show me 100,000 examples of positions, what you decided, and whether it worked out. I'll figure out the pattern myself."

### **What Gets Learned**

You're learning a function that maps:
- **Input:** Current tree state (H_v, G, Var_Q) + position features (board density, game phase, mobility)
- **Output:** Three control signals — how much to trust heuristic, how much budget to allocate, how much to explore

The beauty is the system can discover nonlinear relationships you'd never hand-engineer. Maybe in endgames, Var_Q doesn't matter at all but H_v is critical. Maybe in tactical positions, G is the only signal that matters. The learned controller finds these patterns automatically.

### **The Training Signal Problem**

Here's the tricky part: **How do you know what the "correct" λ should have been for a past position?**

You don't have ground truth. You can't look up the optimal metareasoning policy in a table. So you need to derive it retroactively using one of three approaches:

**Approach 1: Outcome-based learning**
After a game finishes, propagate the outcome backward. Positions in a winning game probably had good λ values. Positions in losing games probably didn't. This is noisy but simple — you're using game results as a proxy for metareasoning quality.

**Approach 2: Efficiency-based learning**
For each position, measure both search cost (simulations used) and decision quality (how good was the chosen move vs the true best). The optimal λ minimizes both. You want to spend few simulations AND pick good moves. This creates a multi-objective optimization problem: balance compute efficiency against decision quality.

**Approach 3: Counterfactual evaluation**
The most expensive but most accurate: After a game, go back to critical positions and replay them with different λ values. See which λ would have led to the best outcome. This is like hindsight — you're asking "knowing how the game turned out, what metareasoning policy would have been optimal?" Then you train toward that policy.

### **Why This Changes Everything**

With a learned controller, you can now:
- **Adapt during training** — early in training when the neural network is weak, maybe trust heuristics more. Later when the network is strong, trust it more. The controller learns this balance.
- **Transfer across positions** — the learned controller discovers which features of a position predict good metareasoning, not just which tree states.
- **Handle new situations** — if the learned controller sees a position type it's never encountered, it can interpolate from similar positions. The formula-based controller just uses the same weights blindly.

### **Success Criteria**

The learned controller succeeds if:
1. It matches or beats the deterministic formula on Reversi positions it was trained on
2. It generalizes to held-out Reversi positions (different game states)
3. When you move to a new game (Chess), it requires less retraining than starting from scratch with a new formula

---

## Phase 6 — Opponent Modeling: Metareasoning Becomes Game Theory

### **The Conceptual Gap**

Your current system treats the game as if it's solitaire. You're deciding "how much compute should I spend to find a good move?" But you're ignoring a crucial fact: **your opponent is also searching, and their search strategy affects what you should do.**

Think about playing against two different opponents:
- **Opponent A:** Novice player, only searches 100 simulations, makes tactical errors
- **Opponent B:** Expert player, searches 10,000 simulations, nearly optimal play

Against Opponent A, you can afford to trust your heuristic more. Why? Because even if your heuristic makes small errors, their blunders will overwhelm your mistakes. You can "get away with" cheaper metareasoning.

Against Opponent B, you need to search deeply. They won't give you free wins. Every small evaluation error matters because they'll exploit it.

**Your current system doesn't distinguish between these cases.** It computes λ based only on tree topology, not on who it's playing.

### **The Game-Theoretic Insight**

Optimal metareasoning in a two-player game is a **nested optimization problem**:
- You're trying to maximize your win probability
- But win probability depends on opponent's strategy
- Your opponent is also doing metareasoning, trying to maximize their win probability
- Their metareasoning affects which positions arise, which affects your optimal λ

This creates a **meta-game** above the object-level game. You're not just playing Reversi against them — you're playing a resource allocation game where you both decide how much compute to invest.

### **What Opponent Modeling Adds**

Track patterns in how your opponent plays:
- **Move selection:** Do they play the most popular moves (suggesting deep search) or make surprising choices (suggesting shallow search or different evaluation)?
- **Time usage:** Are they spending lots of time on simple positions (suggesting they're weak) or making fast, accurate moves (suggesting they're strong)?
- **Style patterns:** Do they play aggressively (lots of captures) or positionally (slow maneuvering)?

From these observations, build a model of opponent strength and style. Then adjust your λ:
- Against weak opponents: increase λ_heuristic (trust your evaluation more, they won't punish small errors)
- Against strong opponents: decrease λ_heuristic (search deeply, can't afford mistakes)
- Against time-pressured opponents: increase your own search budget (they'll make hasty moves you can exploit)

### **The Adaptation Loop**

As the game progresses, you get more data about your opponent. Early moves might suggest they're strong, but then they make a blunder — update your model, increase λ, save compute. This is **online learning during the game.**

The system is answering: "Given what I know about how this opponent thinks, what's the expected value of additional search?"

### **Why This Matters Beyond Games**

In robotics, your "opponent" is the environment. Some environments are predictable (flat floors, no obstacles) — trust your physics model. Some are unpredictable (rough terrain, moving obstacles) — search more.

In LLM inference, your "opponent" is the language task. Some prompts are easy (translation) — trust the draft model. Some are hard (creative writing) — run the full model.

The principle is universal: **Adapt your metareasoning to the difficulty of the problem you're facing.**

### **Success Criteria**

The opponent-modeling system succeeds if:
1. It beats the same AI with non-adaptive metareasoning when playing against weak opponents (by saving compute)
2. It doesn't lose more against strong opponents (maintains quality)
3. Overall, it wins more games per unit of compute spent

---

## Phase 7 — Partial Observability: When You Don't Know What You Don't Know

### **The Fundamental Shift**

Everything so far assumes **perfect information** — you can see the entire game state. H_v, G, and Var_Q measure uncertainty about *which move is best*, not about *what the world state is*.

But most real problems have **hidden state**:
- In StarCraft, you don't know where enemy units are (fog of war)
- In robotics, you don't know exact object positions (sensor noise)
- In military planning, you don't know enemy intentions (intelligence is incomplete)

In these domains, there's a more fundamental question than "how much should I search?" — it's **"should I gather information or execute a plan?"**

### **The Information Gathering Trade-Off**

Imagine a StarCraft scenario:
- You're deciding whether to attack the enemy base
- Your heuristic says "attack now, you'll probably win"
- But you haven't scouted recently — maybe they built defenses

You face a choice:
1. **Execute:** Attack immediately, save the scouting time
2. **Scout:** Send a unit to look, then decide — costs time but reduces uncertainty

This is fundamentally different from the perfect-information case. In Reversi, gathering more information means *running more simulations*. In StarCraft, gathering more information means *sending a scout unit*, which is an action in the world, not a computational decision.

### **Redefining the Topology Signals**

In partial observability, the signals need to measure different things:

**H_v transitions from "visit entropy" to "belief entropy":**
- Perfect info: H_v measures how concentrated the search tree is
- Partial info: H_v measures how uncertain you are about what hidden state you're in

High belief entropy means "I don't know if the enemy built defenses or not" — this directly tells you scouting has high value.

**G transitions from "dominance gap" to "information gain gap":**
- Perfect info: G measures whether one move is clearly best
- Partial info: G measures whether one action reveals more information than others

High information gain gap means "scouting the north entrance reveals much more than the south entrance" — this tells you which scouting action to take.

**Var_Q transitions from "value variance" to "observation variance":**
- Perfect info: Var_Q measures how stable Q-values are across simulations
- Partial info: Var_Q measures how noisy your sensors are

High observation variance means "my lidar readings are inconsistent, probably due to fog" — this tells you to trust prior knowledge over fresh observations.

### **The Meta-Strategy**

The system now makes two-level decisions:

**Level 1: Search or gather?**
- If belief entropy is high (don't know world state), prioritize information gathering
- If belief entropy is low (confident about world state), prioritize planning/search

**Level 2: How to allocate resources?**
- If gathering info: Which sensor to activate? Which scout to send?
- If searching: How much compute to spend? (back to the original λ decision)

### **Why This Is Hard**

In perfect information, all simulation is free (it's just computation). In partial observability, information gathering costs real resources:
- Scouting units can die
- Sensor queries drain battery
- Time spent scouting is time not attacking

So you're balancing **epistemic value** (reducing uncertainty) against **instrumental cost** (resources spent). This is the core problem in active learning, Bayesian experimental design, and partially observable MDPs.

### **Success Criteria**

The partial observability system succeeds if:
1. It scouts more when belief entropy is high (unknown world state)
2. It commits to actions when belief entropy is low (known world state)
3. Overall win rate improves compared to fixed scouting schedules
4. It discovers the optimal exploration-exploitation balance

---

## Phase 8 — Meta-Learning: One Controller to Rule Them All

### **The Universal Metareasoning Dream**

Imagine this: You've now built systems for Reversi, Chess, Go, StarCraft, and robotics path planning. Each one required tuning the λ controller to that domain. Different weights, different thresholds, different position features.

This is unsatisfying. You claim to have discovered universal principles of metareasoning (H_v, G, Var_Q), but you need domain-specific tuning for each application. That suggests the principles aren't actually universal — they're just a useful scaffold that still requires manual engineering.

**The dream:** Train a single controller that works across all domains without per-domain tuning.

### **What Does "Universal" Mean?**

Not that the controller makes identical decisions in Chess and robotics — obviously different domains need different strategies. Rather, it means:

**The controller learns the mapping from topology patterns to optimal resource allocation, and this mapping generalizes.**

For example, it might learn:
- "When H_v is below 0.2 regardless of domain, the decision is clear — commit to the top move"
- "When G exceeds 0.9 in tactical domains (Chess, StarCraft) but Var_Q is high, there's a forcing sequence but it's unstable — search deeply"
- "When both H_v and G are moderate (0.4-0.6), you're in the 'interesting region' where metareasoning matters most"

These patterns should transfer. A high-confidence, low-uncertainty situation in Chess should trigger similar metareasoning as a high-confidence, low-uncertainty situation in robotics.

### **The Meta-Learning Protocol**

Instead of training the controller on one game and hoping it transfers, you train it on multiple games simultaneously:

Train the controller on:
- 10,000 Reversi positions
- 10,000 Chess positions  
- 10,000 Go positions
- 10,000 StarCraft scenarios

With a shared architecture that learns:
- **What's universal:** H_v always indicates uncertainty, G always indicates clarity
- **What's domain-specific:** Chess needs more tactical search, Go needs more strategic evaluation

The controller has two parts:
1. **Domain-invariant encoder:** Maps topology signals (H_v, G, Var_Q) to abstract features
2. **Domain-aware decoder:** Maps abstract features + domain ID to λ values

After training on these five games, you test on a sixth game it's never seen (e.g., Othello). If it performs reasonably without any Othello-specific training, you've proven the metareasoning principles are truly universal.

### **The Adaptation Challenge**

Even with meta-learning, some domain-specific adaptation will be needed. The question is: how much?

**Success looks like:**
- **Zero-shot:** Controller works on new domain with no training (unlikely but ideal)
- **Few-shot:** Controller needs 1,000 examples from new domain to match hand-tuned performance
- **Fast adaptation:** Controller reaches good performance in 10,000 examples, whereas training from scratch needs 100,000

Any of these demonstrate that meta-learning helped. The controller has learned priors that accelerate adaptation.

### **Why This Matters Philosophically**

If you can show that a single metareasoning controller transfers across domains, you've demonstrated that **tree topology signals are universal features of bounded rationality**, not game-specific heuristics.

This moves your work from "clever trick that works in games" to "fundamental principle of computational rationality." That's the difference between 50 citations and 200 citations.

### **Success Criteria**

Meta-learning succeeds if:
1. Single controller trained on 5 games outperforms 5 separately-tuned controllers (showing shared knowledge helps)
2. Controller achieves 60%+ of optimal performance on 6th game with zero examples from that game
3. Controller reaches optimal performance with 10× fewer examples than training from scratch

---

## Phase 9 — Hierarchical Metareasoning: When Should You Think About Thinking?

### **The Meta-Meta Question**

Your current system recomputes λ every 50 simulations. This number (50) was chosen arbitrarily. But think about what this means:

**You're spending computational resources to decide how to allocate computational resources.**

The metareasoning itself costs compute. Every 50 simulations, you:
- Compute H_v (extract all visit counts, calculate entropy)
- Compute G (sort visits, find gap)
- Compute Var_Q (extract all Q-values, calculate variance)
- Run the λ controller (neural network forward pass)

This is cheap compared to a simulation, but it's not free. In some situations, this overhead might be wasteful.

### **The Key Insight**

Some positions have stable topology — the tree structure doesn't change much as you add simulations. In these positions, recomputing λ every 50 simulations is overkill. You could recompute every 200 simulations and get the same λ value.

Other positions have volatile topology — new simulations radically reshape the tree. Maybe one move becomes clearly dominant, or maybe several moves cluster together. In these positions, λ can swing wildly. You might need to recompute every 10 simulations to stay adaptive.

**The question:** How do you know which situation you're in?

### **The Meta-Signal**

Track the rate of change of your topology signals:
- ΔH_v = |H_v(now) - H_v(50 sims ago)|
- ΔG = |G(now) - G(50 sims ago)|  
- ΔVar_Q = |Var_Q(now) - Var_Q(50 sims ago)|

If these deltas are small, the tree is stable — keep using the old λ value. If these deltas are large, the tree is evolving rapidly — recompute λ more frequently.

This is metareasoning about metareasoning. You're deciding when to update your decision about how to allocate compute.

### **The Hierarchical Structure**

You now have three levels:

**Level 0: Object-level search** — running MCTS simulations, exploring the game tree

**Level 1: Metareasoning** — computing λ to decide how to allocate search budget

**Level 2: Meta-metareasoning** — deciding when to recompute λ

Each level observes the level below and makes control decisions. This is a classic hierarchical control system, like how your brain works:
- Low-level: Motor cortex executes movements
- Mid-level: Premotor cortex plans which movements to execute
- High-level: Prefrontal cortex decides when to make a new plan

### **The Resource Trade-Off**

Recomputing λ more frequently:
- **Cost:** More compute spent on metareasoning overhead
- **Benefit:** More adaptive, better resource allocation

Recomputing λ less frequently:
- **Cost:** Might use suboptimal λ in changing situations
- **Benefit:** Lower overhead, more resources for actual search

The optimal frequency depends on tree volatility. Hierarchical metareasoning learns this trade-off.

### **Why This Generalizes**

The principle applies beyond games:

**In robotics:** How often should you replan your trajectory? In stable environments (flat ground), replan every second. In chaotic environments (rocky terrain), replan every 100ms.

**In LLM inference:** How often should you reconsider whether to trust the draft model? For boring text (documentation), check every 50 tokens. For creative writing (poetry), check every 5 tokens.

### **Success Criteria**

Hierarchical metareasoning succeeds if:
1. It reduces metareasoning overhead (fewer λ recomputations) without hurting performance
2. It adapts recomputation frequency based on tree volatility
3. Total compute (search + metareasoning) decreases by 10-20%

---

## Phase 10 — Theoretical Foundations: Proving It's Not Just Empirical

### **Why Theory Matters**

Right now, your system works. You have an 83% win rate with 56% compute savings. That's empirical success.

But you can't answer questions like:
- **Is this optimal?** Could a different metareasoning policy do even better?
- **How much better?** What's the theoretical upper bound on performance?
- **Why does it work?** Is there a principled reason these topology signals matter, or is it just lucky?
- **When will it fail?** Are there problem structures where topology-based metareasoning breaks down?

Without theory, you're engineering. With theory, you're doing science.

### **The Three Core Questions**

**Question 1: Regret Bounds**

Can you prove that your λ controller achieves low regret compared to an oracle that knows the true value of every action?

Regret is: (value of oracle's chosen move) - (value of your chosen move), summed over all decisions.

A good metareasoning system should have regret that grows sublinearly with the number of decisions. This means that over time, you're getting closer to optimal performance, not accumulating errors.

**Conjecture:** ZenoZero achieves regret O(√T·ε) where T is the number of simulations and ε is tree uncertainty (measured by H_v, Var_Q).

If you can prove this, you've shown the system is theoretically sound.

**Question 2: Sample Complexity**

How many simulations does your system need to achieve (1-ε)-optimal play?

In standard MCTS without metareasoning, the answer is O(|A|·ε^(-2)) where |A| is the number of actions. You need to sample proportional to the branching factor.

Your claim is that topology-based metareasoning reduces this. When λ is high (trust heuristic), you need fewer samples.

**Conjecture:** ZenoZero achieves (1-ε)-optimal play with O(|A_effective|·ε^(-2)) samples, where |A_effective| << |A| because soft pruning and early stopping eliminate unpromising branches.

If you can prove this, you've shown the efficiency gains are fundamental, not just empirical.

**Question 3: Information-Theoretic Foundation**

Can you connect H_v, G, and Var_Q to formal information theory concepts?

**Claim:**
- H_v is related to the entropy of the posterior distribution over optimal actions
- G is related to the KL-divergence between the top two actions' value distributions
- Var_Q is related to the mutual information between tree samples and action values

If these connections hold formally, then your topology signals aren't arbitrary — they're measuring fundamental information-theoretic quantities that any rational agent should care about.

### **The Proof Strategy**

Start with a simplified model:
- Assume heuristic is ε-accurate (differs from true value by at most ε)
- Assume tree samples are i.i.d. (not true in MCTS, but a starting point)
- Analyze a single decision (not the full game tree)

Under these assumptions, derive:
1. The expected regret of trusting heuristic (λ=1) vs searching (λ=0)
2. The optimal λ* that minimizes regret + search cost
3. Show that your λ controller approximates λ*

Then relax the assumptions one by one and see how the bounds change.

### **Why This Is Hard**

MCTS is non-stationary (tree changes as you explore), non-i.i.d. (samples are correlated through the tree structure), and involves complex dependencies (parent-child relationships in the tree).

Standard bandit theory assumes i.i.d. samples from fixed distributions. Your problem is more complex.

You'll likely need:
- Concentration inequalities for dependent samples (martingale theory)
- PAC-Bayesian analysis (bounds that hold with high probability)
- Information-theoretic tools (mutual information, KL-divergence)

### **What Success Looks Like**

You don't need to prove tight bounds or handle all edge cases. Even loose bounds are valuable:

**Weak result (still publishable):**
"Under assumption X, Y, Z, ZenoZero achieves regret O(T^0.6)" 
This is worse than optimal O(√T) but proves it's not arbitrarily bad.

**Medium result (good journal paper):**
"For heuristics with bounded error ε, ZenoZero requires O(ε^(-1.5)) samples for (1-ε)-optimal play"
This shows polynomial dependence on error, which is good.

**Strong result (top venue):**
"ZenoZero achieves regret matching information-theoretic lower bounds up to log factors"
This proves it's nearly optimal.

### **Why This Matters**

Theory transforms your work from "we tried this and it worked" to "we can prove this should work, here's why."

Empirical results convince practitioners. Theory convinces researchers. Both are needed for maximum impact.

---

## The Logical Flow Across Phases

Notice how each phase builds conceptually:

**Phase 5 (current):** Hand-engineered metareasoning — you design the λ formula

**Phase 5.5:** Learn the formula — system discovers optimal λ from data

**Phase 6:** Adapt to opponent — metareasoning becomes game-theoretic, accounting for adversarial context

**Phase 7:** Handle uncertainty — extend to partial observability, metareasoning now includes information gathering

**Phase 8:** Generalize across domains — prove the principles are universal, not game-specific

**Phase 9:** Metareasoning about metareasoning — hierarchical control, deciding when to update decisions

**Phase 10:** Prove it's optimal — theoretical analysis shows why this works and what the limits are

Each phase asks a deeper question about computational rationality:
- 5: How should I allocate resources?
- 5.5: Can I learn to allocate resources?
- 6: How does adversarial context affect allocation?
- 7: Should I gather info or act?
- 8: Do these principles generalize?
- 9: When should I reconsider my allocation strategy?
- 10: Why does this work and when does it fail?

This progression takes you from engineering (Phase 5) to science (Phase 10).