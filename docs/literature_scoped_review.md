<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# I am investigating whether the following idea has prior art in academic literature (AI, reinforcement learning, search algorithms, or game AI):

A Monte Carlo Tree Search (MCTS) system that:
Computes tree topology metrics during search:
Visit distribution entropy (normalized Shannon entropy of child visit counts)
Dominance gap between top two actions
Variance of Q-values across children
Uses these metrics as a real-time meta-control signal to:
Dynamically weight heuristic injection into UCB
Adapt the exploration coefficient (c_puct)
Adapt the simulation budget per move
Trigger early stopping when structural convergence is detected
Optionally logs these topology signals and trains a learned meta-controller
(small MLP) to predict optimal search parameter settings from tree geometry.
The key distinction:
This is NOT:
Standard adaptive UCB tuning
Static entropy bonuses
Simple regret-based sampling
Offline hyperparameter tuning
Variance reduction techniques
POMDP belief entropy planning
It IS:
A multi-signal, runtime meta-control system where tree geometry directly controls multiple search parameters within a single MCTS instance.
Please:
Identify any existing papers that:
Use visit entropy of the MCTS tree as a control variable
Use dominance gap or spectral separation as a stopping or control criterion
Use value variance across children as a search adaptation signal
Dynamically adapt search budget within a single move based on tree statistics
Learn a controller that maps tree statistics to MCTS parameters
Clearly distinguish between:
Entropy used for exploration regularization
Variance reduction techniques
Adaptive MCTS parameter tuning
True topology-aware meta-control systems
Focus on peer-reviewed papers from 2015–2025 in:
NeurIPS, ICML, ICLR, AAAI, IJCAI
IEEE Transactions on Games
JAIR
arXiv (if relevant and recent)
If no paper matches this full system, identify:
Closest partial matches
Missing components
Clear research gap
Summarize:
Whether this appears novel
How it differs from adaptive MCTS and information-theoretic MCTS
Whether similar ideas exist under a different terminology
Be precise and citation-heavy.

Your proposed system appears novel as a *single*, multi-signal, topology-aware meta-controller for MCTS; existing work covers fragments (adaptive parameters, entropy/variance use, adaptive budgets, learned controllers), but not your full combination where root/branch geometry jointly drives several search parameters online within one MCTS instance.[^1][^2][^3][^4][^5][^6][^7]

Below I’ll structure the answer around what *does* exist, what is missing, and where the research gap seems to be.

## 1. Adaptive / self-tuning MCTS parameters

There is a line of work on **self-adaptive / online parameter tuning**, but it does not use explicit tree-topology metrics like visit-entropy or dominance gaps as control signals.

- Winands et al., “Self-Adaptive Monte Carlo Tree Search in General Game Playing,” IEEE Transactions on Games, 2020 (original conference version around 2018) propose SA‑MCTS, which tunes search-control parameters (e.g., exploration constant, playout-related parameters) online per game.[^5][^6][^8]
    - Parameters are treated as an arm in an online tuner (bandit / EA-style allocation over parameter settings), *not* driven by structural descriptors such as entropy of the child-visit distribution or variance of Q across children.[^6]
    - The tuner allocates samples to parameter configurations and selects those that yield higher performance; the statistics are on *parameter configurations vs. win rate*, not on intrinsic tree-geometry features.[^6]
- Various surveys (e.g., Browne et al., “A Survey of Monte Carlo Tree Search Methods”) catalog parameter tuning and adaptive variants, but again treat parameters as objects of tuning instead of outputs of a learned mapping from topology features.[^7]

So: **adaptive MCTS parameter tuning exists, but not as a direct function of normalized visit entropy, dominance gap, or Q-variance at a node.**[^5][^7][^6]

## 2. Entropy in/around MCTS

### 2.1 Entropy for exploration / regularization

- “Maximum Entropy Monte-Carlo Planning” (Zhang \& Chen, NeurIPS 2019) introduces MENTS, which augments MCTS with maximum-entropy policy optimization and evaluates each node using softmax value estimates.[^9][^10][^4][^1]
    - Entropy here is internal to the *policy optimization objective* (soft value backup, Boltzmann exploration), not an explicit “visit-distribution entropy at the root” used as a separate control signal.[^4][^1]
    - It regularizes policies to encourage exploration and improve regret bounds, but does not drive dynamic changes to cpuct or simulation budget based on the *shape* of the visit histogram.
- MCTS with Boltzmann Exploration (NeurIPS 2023) similarly integrates Boltzmann-type exploration, effectively using temperature-controlled stochasticity, but again not an explicit topological entropy metric used for meta-control.[^10]
- Cross-entropy methods for parameter search (mentioned in SA‑MCTS and related work) use cross-entropy in the *parameter space* of algorithms, not the entropy of child visits in a given tree.[^6]

Thus: **entropy is used as an exploration regularizer or objective in policy/value learning, not as a runtime “tree-geometry metric” that controls multiple MCTS parameters.**[^1][^10][^4][^6]

### 2.2 Tree-entropy as explicit control

I did not find peer-reviewed NeurIPS/ICML/ICLR/AAAI/IJCAI/IEEE T-Games/JAIR papers that:

- explicitly define “normalized Shannon entropy of child visit counts” at a node and
- use that *explicitly* to adapt cpuct, simulation budget, or heuristic injection strength within a single MCTS run.

Entropy is present as a concept in information-theoretic planning and mutual-information-based search, but those works typically operate in belief space or over action-value distributions in an information-gain sense (e.g., POMDP belief entropy), not as your tree-topology meta-signal.[^11][^12]

## 3. Variance and uncertainty in MCTS

There is more on variance, but again mainly as a **value estimator design**, not as a global topology-aware controller.

- Lanctot et al., “Variance Reduction in Monte-Carlo Tree Search” (NIPS workshop / tech report) explicitly analyses variance of returns and proposes control variates and related techniques to reduce variance in value estimates.[^2]
    - Variance here is about improving the estimate of a node’s value, not about controlling cpuct, simulation budget, or heuristic mixture based on Q-variance across children.
- Ruzicka et al., “Utilizing Variance and Uncertainty in Monte-Carlo Tree Search” (tech report / workshop) incorporate variance and uncertainty into the selection rule, adding a factor that shifts the balance between Q and U based on uncertainty.[^3]
    - This is essentially a refined UCB/PUCT-like selection formula where value variance affects which child is chosen, i.e., **local selection** rather than **global meta-control over search parameters**.
- Some adaptive MCTS/search papers in recent years (e.g., “From Static to Dynamic: Adaptive Monte Carlo Search” – AMCS) adjust exploration weight or path expansion based on uncertainty or variance measures, but again at the selection/backup level, not via an explicit controller that takes multiple topology statistics (entropy, dominance gap, variance) and outputs a vector of parameter settings.[^13][^14]

So: **value variance is used for improved selection / estimation (variance reduction, uncertainty-aware UCB), but I found no work where Q-variance over children is aggregated into a “tree geometry” vector that controls cpuct, heuristic mixing, and per-move budget simultaneously.**[^14][^2][^3]

## 4. Dominance gap, spectral separation, and stopping criteria

- General MCTS surveys mention “robust child”, “max-robust child”, and stopping rules like “search until visit count exceeds a threshold” or “until best child is sufficiently ahead,” but these are usually heuristic thresholds, not framed as spectral/dominance-gap metrics with theoretical grounding.[^7]
    - E.g., the survey discusses rules such as selecting a “max-robust child” that has both highest visit count and highest reward, and stopping once an acceptable visit count is achieved.[^7]
- I did not find a mainstream 2015–2025 paper in the venues you listed that uses something like:
    - “gap between top two child visit counts or Q-values”
    - as an explicit *stopping* or *meta-control* criterion with a well-defined dominance-gap metric, or
    - a “spectral separation” of the visit/Q distribution as a control signal.

There are regret-minimization and best-arm identification works in bandits that use gaps between arms for stopping, but their integration into full MCTS as a “tree topology gap” for controlling search parameters within a move does not seem to appear in the standard literature.[^7]

So: **dominance-gap-based stopping is conceptually close to best-arm identification, but I do not see it instantiated as a general topology-aware meta-controller for MCTS.**[^7]

## 5. Dynamic simulation budget within a move

There are several lines where simulation budgets are **adapted or allocated**, but typically at a coarser granularity or using performance statistics rather than tree-geometry measures.

- SA‑MCTS and related self-adaptive methods allocate simulations among parameter settings, which indirectly changes how many rollouts go through certain configurations, but this is a *meta-bandit over configurations*, not a per-move, per-node budget driven by entropy/variance.[^5][^6]
- Some adaptive MCTS schemes (including AMCS-style approaches) adapt path length, exploration strength, or evaluate different rollout depths as functions of state characteristics or remaining horizon, not of root visit-entropy or Q-variance across children.[^13][^14]
- I did not find a widely cited paper that, within a single decision step:
    - monitors evolving root or branch statistics (e.g., entropy dropping, dominance gap increasing) and
    - uses them to *stop early* or *increase budget* for that move in a principled, topology-aware way.

Heuristics like “stop when visit count > N or time > T” are common, but those are not topology-aware in your sense.[^7]

Thus: **dynamic budget adaptation exists in terms of time-based cutoffs and configuration tuning, but not as a function of a multi-dimensional topology vector (entropy, dominance gap, Q-variance) for a single MCTS instance.**[^5][^6][^7]

## 6. Learned controllers mapping tree statistics → parameters

Here we get closest, but still not to your exact setup.

- SA‑MCTS and its extensions can be interpreted as a *meta-agent* (bandit/EA) that, based on past performance statistics, chooses parameter settings for MCTS online.[^6][^5]
    - However, the input features to this meta-agent are performance/win-rate statistics over parameter configurations, not explicit tree-geometry features like visit entropies or Q-variance.
- Algorithm portfolio approaches (e.g., Mendes et al. referenced in the CoG survey piece) use a meta-agent to select among different search algorithms for a given game instance.[^15]
    - Again, features tend to be game descriptors or performance statistics, not direct measures of tree topology in the current search.
- In LLM planning contexts (e.g., recent adaptive Monte Carlo search frameworks) there are controllers that adapt cpuct or exploration bonuses based on progress or uncertainty, but I did not see an explicit *MLP that takes a vector of topology statistics from the current tree and outputs parameters for the same ongoing search*. Recent AMCS-like frameworks primarily adjust internal evaluation or path expansion policies rather than building a separate topology-driven controller.[^14][^13]

So: **learned meta-controllers for search exist, but the specific idea “log entropy/gap/variance features and train a small network to output cpuct, heuristic weights, and budget for the same MCTS instance” does not seem to appear in standard venues.**[^15][^5][^6]

## 7. Distinguishing concept classes

Given your request, here is a concise conceptual separation, with examples.

- **Entropy used for exploration regularization**
    - MENTS (Maximum Entropy Monte-Carlo Planning) and Boltzmann-Exploration MCTS incorporate entropy into the *policy objective* or sampling distribution to encourage exploration and improve regret/sample-efficiency.[^10][^4][^1]
    - Entropy is inside the value/policy computation, not a separate meta-signal controlling cpuct, budget, or heuristic mixing based on tree geometry.
- **Variance reduction techniques**
    - Lanctot’s variance-reduction work uses control variates and related estimators to reduce variance of value estimates at nodes.[^2]
    - Some later work uses variance/uncertainty to refine selection rules (e.g., Ruzicka’s uncertainty factors), but these stay at the level of *node evaluation* and *action selection*, not global meta-control.[^3]
- **Adaptive MCTS parameter tuning**
    - SA‑MCTS and related work perform online tuning of parameters like exploration constants and rollout settings using general-purpose tuners (bandits, EAs, cross-entropy methods).[^5][^6]
    - Adaptation is typically keyed to performance/win-rate or offline training, not to instantaneous measures of tree topology such as visit-entropy or dominance gaps at the root.
- **Topology-aware meta-control systems (your category)**
    - Would explicitly compute structural descriptors of the *current* search tree (e.g., entropy of visit counts, dominance gaps, Q-variance across children, depth-wise branching statistics) and map them to:
        - cpuct (or equivalent exploration weight),
        - heuristic injection weights in UCB/PUCT,
        - per-move simulation budget / early stopping, and
        - possibly other knobs (rollout policy choice, depth limits).
    - This **multi-signal, multi-parameter, single-instance** control system is not present as a full design in the literature I could locate.[^1][^2][^3][^6][^5]


## 8. Closest partial matches and missing components

Putting this together:

- **Closest lines of work:**
    - Self-adaptive MCTS (Winands et al., IEEE T-Games 2020): online parameter tuning within MCTS using statistical tuning methods.[^6][^5]
    - Maximum-entropy / Boltzmann MCTS (MENTS, Boltzmann exploration): uses entropy in the value/policy objective to shape exploration.[^4][^10][^1]
    - Variance/uncertainty-aware MCTS (Lanctot; Ruzicka et al.): use variance to improve estimates/selection, not to control global parameters.[^2][^3]
    - Adaptive Monte Carlo Search (AMCS-like frameworks): dynamic exploration bonuses and path expansion, but more focused on adaptive evaluation than explicit topology-aware control.[^13][^14]
- **Missing components relative to your proposal:**
    - Explicit **normalized visit-entropy** over children as a control variable for cpuct, heuristic mixing, and per-move budget.
    - **Dominance gap / spectral separation** (e.g., between top two actions or eigenstructure of visit/Q distributions) as a stopping/early-convergence signal.
    - **Variance of Q-values across children** explicitly used as a meta-signal, not just as part of node-level uncertainty.
    - A **single learned meta-controller (MLP)** that, given a feature vector of topology statistics from the ongoing tree, outputs:
        - cpuct,
        - heuristic weights inside UCB/PUCT,
        - simulation-budget adjustments or early-stop decisions,
        - potentially other continuous settings, *within the same MCTS instance and at runtime*.

This combination of multi-signal topology metrics plus multi-parameter control within a single search run does not appear to be instantiated in the cited literature.[^3][^1][^2][^5][^6][^7]

## 9. Novelty assessment and relation to existing ideas

- **Novelty:**
    - In terms of *individual ingredients*, you are building on known ideas: adaptive parameter tuning, entropy-based exploration, variance- and uncertainty-aware selection, and learned meta-controllers.[^1][^2][^3][^5][^6]
    - What appears novel is the *integration*: using a set of **tree-topology metrics (visit-entropy, dominance gap, Q-variance)** as a **shared, real-time state** for a **meta-controller** that simultaneously manages *multiple* MCTS control knobs (cpuct, heuristic mixing, simulation budget, early stopping) within a single move. I did not find a prior work that matches this full description in the 2015–2025 venues you specified.[^1][^5][^6][^7]
- **Difference from adaptive MCTS:**
    - Adaptive MCTS typically tunes parameters based on performance statistics or generic tuners over parameter configurations, often per game or per class of instances, and not directly as a function of interpretable topology metrics.[^5][^6][^7]
    - Your proposal moves adaptation into a *stateful control loop* where the “state” is the geometry of the current tree and the “action” is a vector of search parameters for that same tree.
- **Difference from information-theoretic MCTS:**
    - Information-theoretic approaches, including maximum-entropy planning and mutual-information-based search, embed entropy/MI in the *objective*, e.g., soft value backups, information gain, or belief-space entropy reduction.[^11][^4][^1]
    - You instead treat entropy and related measures as *observed signals* from the current search process used to control algorithmic knobs, not as terms in the reward/objective itself.
- **Possible hidden relatives under other terminology:**
    - Work on “meta-MCTS” or “hyper-heuristics for search” sometimes discusses controllers that select among search strategies, but these generally operate at the algorithm-selection level rather than as a continuous topology-to-parameter mapping.[^15]
    - Some planning and LLM-reasoning frameworks using MCTS integrate “search progress” metrics (e.g., number of high-value nodes, depth distributions), but these are more like heuristic stopping criteria than a full meta-control architecture.

Given the above, your proposal looks like a **plausible research gap**: a topology-aware, multi-signal, learned meta-controller that continuously maps tree geometry to MCTS control parameters within a single search episode. Formalizing the feature set, the control policy class (e.g., small MLP), and evaluation on standard benchmarks (Go-like games, GVGP, or LLM reasoning tasks) would make a strong, clearly differentiable contribution relative to existing adaptive and information-theoretic MCTS work.[^2][^3][^1][^6][^5][^7]
<span style="display:none">[^16][^17][^18][^19][^20][^21][^22][^23][^24][^25][^26][^27][^28][^29]</span>

<div align="center">⁂</div>

[^1]: http://papers.neurips.cc/paper/9148-maximum-entropy-monte-carlo-planning

[^2]: https://mlanctot.info/files/papers/nips11vrmcts.pdf

[^3]: https://ml-research.github.io/papers/ruzicka2023utilizing.pdf

[^4]: https://proceedings.neurips.cc/paper_files/paper/2019/file/7ffb4e0ece07869880d51662a2234143-Paper.pdf

[^5]: https://ieeexplore.ieee.org/document/8556454/

[^6]: https://dke.maastrichtuniversity.nl/m.winands/documents/New-AdaptiveMCTSforGGP.pdf

[^7]: http://www.incompleteideas.net/609 dropbox/other readings and resources/MCTS-survey.pdf

[^8]: https://www.semanticscholar.org/paper/e8a146517171a0b9f34d4a5d9d0f832c4a03bae7

[^9]: https://rbcborealis.com/publications/maximum-entropy-monte-carlo-planning/

[^10]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/f670ef96387d9a5a8a51e2ed80cb148d-Abstract-Conference.html

[^11]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12294082/

[^12]: https://www.cwi.nl/documents/195389/talk(1).pdf

[^13]: https://arxiv.org/pdf/2509.24351.pdf

[^14]: https://openreview.net/pdf/d8097869400fabd22970495a9c89832ae5ed20c1.pdf

[^15]: https://ieee-cog.org/2021/assets/papers/paper_320.pdf

[^16]: https://arxiv.org/html/2502.04751v1

[^17]: https://hexdocs.pm/macula_neuroevolution/meta-controller.html

[^18]: https://arxiv.org/abs/2510.03632

[^19]: https://arxiv.org/pdf/2210.06352.pdf

[^20]: https://www.sciencedirect.com/science/article/abs/pii/S0952197624003105

[^21]: https://instadeep.com/research/paper/alphanpi-x-learning-compositional-neural-programs-for-continuous-control/

[^22]: https://scholars.ln.edu.hk/en/publications/self-adaptive-monte-carlo-tree-search-in-general-game-playing/

[^23]: https://en.wikipedia.org/wiki/Monte_Carlo_tree_search

[^24]: https://arxiv.org/pdf/2509.26435.pdf

[^25]: https://cris.maastrichtuniversity.nl/files/90048893/Winands_2020_Self_adaptive_rolling_horizon_evoluationary_algorithms.pdf

[^26]: https://www.geeksforgeeks.org/machine-learning/monte-carlo-tree-search-mcts-in-machine-learning/

[^27]: https://arxiv.org/pdf/2511.06142.pdf

[^28]: https://courses.cs.washington.edu/courses/cse599i/18wi/resources/lecture19/lecture19.pdf

[^29]: https://project.dke.maastrichtuniversity.nl/games/files/phd/Chaslot_thesis.pdf

