# Runner-up cap on the official NPC runs

Start from the 105 verified fits of the released Netta NPC pipeline on CIFAR-100 BYOL. Each row retains the official acquisition order, observed labels, final AUM mask, and 10,000-image test set. The old graph pilot used a 45,000-image approximation of NPC acquisition and usually voted with all acquired labels. The current test uses the official 50,000-image NPC run and its final AUM keep set. [Released NPC setup and pseudocode](../netta/PROTOCOL.md)

## Step by step

1. Run the released acquisition and LowBudgetAUM. Denote the final retained acquired images by $K$. Their observed labels are $\tilde y_i$.
2. For the released predictor, train the 500-epoch linear classifier on $K$ and predict $\arg\max_c (Wz_x+b)_c$.
3. **For the graph predictor, connect each train image to its 13 nearest other train images by cosine similarity. Connect each test image to its 13 nearest train images.** Let $A$ be the train adjacency and $T$ the test-to-train adjacency. Count all walks with six edges ending at each retained annotation.

$$
h_i(x)=(TA^5)_{xi},\qquad i\in K.
$$

4. Raw graph voting gives annotation $i$ weight $h_i(x)$. **Capped voting gives it weight $w_i(x)=\min\{h_i(x),s_2(x)\}$, where $s_2(x)$ is the second-largest positive count among distinct retained images.** When fewer than two counts are positive, use $s_2(x)=+\infty$. Either graph arm adds weights by observed class. Ties use the lowest class index.

$$
\hat y(x)=\arg\max_c \sum_{i\in K:\tilde y_i=c} w_i(x).
$$

For counts $(16,4,3)$ with observed labels $(\text{cat},\text{dog},\text{dog})$, raw votes choose cat $16$ to $7$. The cap gives counts $(4,4,3)$ and chooses dog $7$ to $4$. With at least two reachable annotations, at least two sources have the maximum capped weight. Each individual source therefore supplies at most half the total weight.

| Component | Released NPC | Raw graph control | Runner-up cap |
|---|---|---|---|
| Acquisition and final label set | Official NPC and AUM keep set $K$ | Same | Same |
| Predictor | Trained linear logits | Six-edge graph votes | Six-edge graph votes |
| Weight of one source | Enters the fitted model through supervised training | Its walk count | Its walk count capped at the runner-up |

Cap versus raw isolates the cap. Cap versus released NPC measures the whole predictor replacement. All settings use the original saved per-episode AUM masks, equal label budgets, the same features, the same noise draw, and the same held-out images. Seeds 71–73 and budgets 400, 800, 1,200, 1,600, and 2,000 cover clean, symmetric 30/40/50%, and cyclic asymmetric 30/40/50%.

## Two other one-weight rules

After the cap replay, run both rules on the same official episode snapshots. Treat this as an exploratory comparison on the existing three seeds. The replay checks that every raw test count reproduces before adding the variants.

- **Square-root votes.** Set $w_i(x)=\sqrt{h_i(x)}$. Walks ending at one annotation have diminishing returns.
- **AUM-rank votes.** Set $w_i(x)=h_i(x)\omega_i$. Among the accepted sources with the same observed class $c$, let $r_i$ be the zero-based AUM midrank and $n_c$ the number of accepted sources. Set $\omega_i=\tfrac12+(r_i+\tfrac12)/n_c$. The mean multiplier is one in each observed class. With equal probability, choose a uniform source or compare two uniform sources and take the higher AUM score. Multiplying the resulting selection probability by $n_c$ gives $\omega_i$.

The aggregation and test prediction are the same as the raw six-hop control.
