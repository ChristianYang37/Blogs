---
title: "Reasoning as Optimization: A Rate for Test-Time Scaling"
date: 2026-05-25T02:00:00+08:00
math: true
---


**The setup.** Thinking LLMs spend extra compute between `<think>` and `</think>` tokens before emitting an answer; pass@1 climbs with thinking length. This was a launch headline when OpenAI o1 and DeepSeek R1 shipped; by the time GPT-5.5, Opus 4.7, Gemini 3.5, Qwen 3.6, and Kimi K2.5 arrived, the curve had become table stakes — every frontier model has it, nobody markets it anymore ([Snell et al. 2024](https://arxiv.org/abs/2408.03314); [OpenAI 2024](https://arxiv.org/abs/2412.16720); [DeepSeek 2025](https://arxiv.org/abs/2501.12948); [Muennighoff et al. 2025](https://arxiv.org/abs/2501.19393)). The mechanistic question — *what's the rate, and what determines it* — is less settled. This post adds one angle.

**The question.** At one layer and one head of the transformer, what does the attention output at the `</think>` position do as the thinking trajectory grows?

**The conceptual answer.** It runs a weighted vote. Each reasoning token contributes one more vote to an average over the value vectors the model has emitted. Good votes — tokens encoding answer-relevant content — pull the average toward the right answer; bad votes pull it away. If the model has been trained so that good votes are cast with some baseline probability and reliably score high in attention, the average converges to a small neighborhood of the correct answer's embedding, and the probability of landing outside that neighborhood decays exponentially in the number of reasoning steps.

The rest of this post unpacks that answer line by line: where the running average comes from, what makes a token "good," what conditions make the exponential rate kick in, and where the rate constant comes from.

## Where this sits

Two well-established framings of test-time reasoning are worth naming up front.

The **CoT-as-multi-step-gradient-descent** line ([von Oswald et al. 2022](https://arxiv.org/abs/2212.07677); [Akyürek et al. 2022](https://arxiv.org/abs/2211.15661); [Cheng et al. ICLR 2025](https://arxiv.org/abs/2502.21212); [multi-head 2025](https://arxiv.org/abs/2508.08222)) shows that a linear self-attention layer with appropriate weights implements one gradient descent step on an in-context regression loss, and that chain-of-thought lifts a one-layer transformer from one-shot regression to provably correct multi-step optimization. This is the cleanest existing answer to "reasoning is optimization." It is also restricted to synthetic in-context tasks at training time, not the decode-time dynamics of real thinking models.

The **scaling-by-aggregation** line ([Wang et al. 2022](https://arxiv.org/abs/2203.11171); [Yao et al. 2023](https://arxiv.org/abs/2305.10601); [Chen et al. NeurIPS 2025](https://arxiv.org/abs/2411.19477); [Kim et al. 2025](https://arxiv.org/abs/2510.17472)) proves exponential or power-law decay of failure probability in test-time compute, but via aggregation across *parallel* samples rather than within a single chain.

Two more recent threads sit closer to the dynamics I want to look at. The **looped / latent-recurrence** camp ([Saunshi et al. ICLR 2025](https://arxiv.org/abs/2502.17416); [Geiping et al. 2025](https://arxiv.org/abs/2502.05171)) identifies CoT steps with iterated applications of a single transformer block and reports convergent hidden-state dynamics. The **stochastic-process** camp ([Bondaschi et al. 2026](https://arxiv.org/abs/2603.00306); [Santilli et al. 2025](https://arxiv.org/abs/2506.04374); [Oh 2026](https://arxiv.org/abs/2601.15482)) treats the reasoning trajectory as a probabilistic process.

The closest *empirical* neighbor to what I'll discuss is [Choi et al. 2025](https://arxiv.org/abs/2509.26522) ("Entropy After </Think>"): they monitor next-token entropy at the `</think>` position throughout thinking and see a trajectory that decreases and stabilizes as thinking grows. The plateau they threshold for early exit is the empirical signature of a converging running average. [Bogdan et al. 2025](https://arxiv.org/abs/2506.19143) ("Thought Anchors") goes one step deeper mechanistically: specialized attention heads broadcast from later sentences back to a small set of high-importance prefix tokens — exactly the configuration a running average converges *to*.

The angle I'm taking is at the **attention-output level**: fix the `</think>` position, look at what softmax attention computes there as new reasoning tokens arrive, and ask what the dynamics are. The mathematical object — a softmax-weighted running mean over emitted value vectors — and the rate — exponential in the reasoning horizon under an anchoring assumption — are, as far as I can find, not explicitly stated in the literature.

## The plan, in one paragraph

We will look at a single layer $i$ and a single attention head $h$ of the transformer. We treat the position of `</think>` as fixed, even though that token has not actually been emitted yet; we just track what the attention output *would be* at that position throughout the thinking process. By the algebra of softmax attention, this output evolves as a weighted running average over the value vectors of emitted tokens. The dynamics question then becomes: when does this running average converge to a vector that decodes to the correct answer? Five conditions on the LLM — each measurable from inference behavior alone — turn out to be enough. The convergence happens at an exponential rate in the number of reasoning tokens, and the rate is set by the model's anchor-emission probability.

## Step 1: what an attention head computes

At any position $p$ in a transformer, an attention head computes

$$\mathrm{Attn}(q_p, K, V) \;=\; \sum_{k} \mathrm{softmax}(q_p^\top k_k) \cdot V_k \;=\; \sum_{k} \frac{\exp(q_p^\top k_k)}{\sum_l \exp(q_p^\top k_l)}\, V_k,$$

where the sum runs over keys $k_k$ in the context, $q_p$ is the query at position $p$, and $V_k$ are the value vectors. The output is a convex combination of value vectors: the softmax weights are non-negative and sum to one.

We are going to pick $p$ to be the position of `</think>`. We write $x_{i,h,T} \in \mathbb{R}^d$ for the layer-$i$, head-$h$ attention output at the `</think>` position after the first $T$ reasoning tokens have been emitted between `<think>` and `</think>`.

A subtlety: the `</think>` token has not yet been emitted; it gets emitted last, after the model finishes thinking. We are treating its position as a *virtual* anchor — we ask what the attention head *would compute* there, conditional on the prefix so far. The query at the `</think>` position is well-defined as soon as the layer-$(i-1)$ output for that position is materialized (which happens during the forward pass). So this is an analysis device, not a claim about an actually decoded answer.

For the rest of this post we suppress the $i, h$ subscripts and just write $x_T$. Everything below is for one fixed (arbitrary) choice of layer and head.

## Step 2: what changes when a new reasoning token is generated

Now suppose we go from $j-1$ reasoning tokens to $j$ reasoning tokens — the model has just emitted one more token between `<think>` and `</think>`. The set of keys grows by one. What happens to the attention output?

Let me set up notation. Write $q$ for the query at the `</think>` position (we drop subscripts; everything below is for one fixed layer/head). Write $k_k$ and $V_k$ for the key and value of the $k$-th emitted token, $k = 1, \ldots, j$. The softmax denominator after $j$ tokens is

$$s_j \;:=\; \sum_{k=1}^{j} \exp(q^\top k_k).$$

Adding the $j$-th token adds one more term:

$$s_j \;=\; s_{j-1} \;+\; \exp(q^\top k_j).$$

The attention output after $j$ tokens is

$$x_j \;=\; \sum_{k=1}^{j} \frac{\exp(q^\top k_k)}{s_j}\, V_k.$$

This is *not* the same as $x_{j-1}$ rescaled. The new token enters, but more importantly, *all the existing weights shrink* because the denominator $s_j$ is bigger than $s_{j-1}$. Specifically, each old weight gets multiplied by $s_{j-1}/s_j < 1$:

$$\frac{\exp(q^\top k_k)}{s_j} \;=\; \frac{s_{j-1}}{s_j} \cdot \frac{\exp(q^\top k_k)}{s_{j-1}}, \qquad k \le j-1.$$

(Just multiply numerator and denominator by $s_{j-1}$.) Pulling this through the sum, we can split $x_j$ into a contribution from old tokens (rescaled) and a contribution from the new token:

$$x_j \;=\; \frac{s_{j-1}}{s_j} \underbrace{\sum_{k=1}^{j-1} \frac{\exp(q^\top k_k)}{s_{j-1}}\, V_k}_{=\, x_{j-1}} \;+\; \frac{\exp(q^\top k_j)}{s_j}\, V_j \;=\; \frac{s_{j-1}}{s_j}\, x_{j-1} \;+\; \frac{\exp(q^\top k_j)}{s_j}\, V_j.$$

Let me write this more compactly. Define the "decay factor"

$$\lambda_j \;:=\; 1 - \frac{s_{j-1}}{s_j} \;=\; \frac{\exp(q^\top k_j)}{s_j} \;\in\; (0, 1),$$

and the "step" contributed by the new token

$$g_j \;:=\; \frac{\exp(q^\top k_j)}{s_j}\, V_j \;=\; \lambda_j \, V_j.$$

Then

$$\boxed{\;x_j \;=\; (1 - \lambda_j)\, x_{j-1} \;+\; g_j.\;}$$

This is the one-step recurrence we are going to spend the rest of the post analyzing.

## An aside: is this SGD?

The form $x_j = (1 - \lambda_j) x_{j-1} + g_j$ is, line by line, identical to stochastic gradient descent with decreasing weight decay: $\lambda_j$ is the decay rate, $g_j$ is the step.

This is where "reasoning is optimization" comes from as a slogan. It is also where the slogan starts to mislead. The variable $g_j$ in standard SGD is the (negative) gradient of an explicit loss function $L$, scaled by a step size: $g_j = -\eta \nabla L(x_{j-1}, \text{data}_j)$. Here $g_j = \lambda_j V_j$ — a re-weighted value vector, with no explicit loss attached.

We could *postulate* a loss function $L$ such that $g_j$ is its stochastic gradient. But we cannot read this $L$ off any inference observation; it would be an axiom we layer on top of the model and then ask the reader to accept. The proof I am about to walk through *does not* postulate any such $L$. It uses the recurrence as-is, treating $V_j$ as a value vector the model emits — not as a gradient of anything.

So: the recurrence has the algebraic form of SGD. The reading of it as actual optimization is rhetorical. That distinction matters for what we can claim and what we can prove.

## Step 3: unrolling the recurrence

Let us unroll $x_j = (1 - \lambda_j) x_{j-1} + g_j$ all the way back to $x_0$.

The product of decay factors $\prod_{i=k+1}^{j} (1 - \lambda_i) = \prod_{i=k+1}^{j} (s_{i-1}/s_i)$ telescopes:

$$\prod_{i=k+1}^{j} \frac{s_{i-1}}{s_i} \;=\; \frac{s_k}{s_j}.$$

So the contribution of $g_k$ to $x_j$ is $(s_k/s_j) \cdot g_k$. Substituting back $g_k = \lambda_k V_k = (\exp(q^\top k_k)/s_k) V_k$,

$$\frac{s_k}{s_j} \cdot g_k \;=\; \frac{s_k}{s_j} \cdot \frac{\exp(q^\top k_k)}{s_k}\, V_k \;=\; \frac{\exp(q^\top k_k)}{s_j}\, V_k.$$

This is just the softmax weight on the $k$-th token as seen from time $j$. Let's name it:

$$w_{j,k} \;:=\; \frac{\exp(q^\top k_k)}{s_j}, \qquad k = 1, \ldots, j.$$

Then summing over $k$,

$$\boxed{\;x_j \;=\; \sum_{k=1}^{j} w_{j,k}\, V_k, \qquad w_{j,k} \ge 0, \quad \sum_{k=1}^{j} w_{j,k} = 1.\;}$$

This is the **running-average** form. The attention output at `</think>`, after $j$ reasoning tokens have been emitted, is literally a softmax-weighted average over the value vectors of those tokens.

Strictly speaking, this is just the original attention formula re-derived. But the recurrence has told us something the formula alone did not: *how the running average evolves*. Each new token slightly shrinks the contribution of all previous tokens (by the factor $s_{j-1}/s_j$) and adds its own.

## Where does it converge?

Now the dynamics question: as $j \to \infty$, where does $x_j$ go?

If we had random $V_k$ with bounded norm and well-behaved expectations, the running average would concentrate around the mean of the value distribution by standard concentration arguments. But this is not what we have. The value vectors are determined by the trained LLM, conditional on the question $Q$ and the trajectory of emitted tokens. There is no neat i.i.d. structure.

The intuition needs to be more specific. For the running average to converge to a "correct-answer embedding," two ingredients have to be present:

1. **Some tokens in the vocabulary have value vectors pointing near the right answer.** Without this, no weighted average can produce a useful direction.
2. **The model emits those tokens often enough, and weights them highly enough, that they dominate the average.**

These are the two ingredients we will turn into formal assumptions next.

## The five conditions, with intuition

I'll state each condition informally first, then give the formal version, then say why it's reasonable.

### Condition 1: Anchor accuracy

*Informally*: For each question $Q$, some tokens in the vocabulary are "answer-bearing" — their value vectors at our chosen layer/head point near a target $V^*(Q)$.

*Formally*: There is a question-conditional subset $\mathcal{A}(Q) \subseteq \mathrm{vocab}$ — the *anchor set* — and a target $V^*(Q) \in \mathbb{R}^d$, such that for every $a \in \mathcal{A}(Q)$,

$$\|V(k_a) - V^*(Q)\| \;\le\; \varepsilon_{\mathrm{anc}}.$$

*Why this is reasonable*: A trained LLM, given a math question, has learned that certain tokens (digits or symbols of the answer, or pre-cursors like "= 4") are answer-bearing. After the value projection at the chosen layer/head, their embeddings should encode answer-relevant information. The constant $\varepsilon_{\mathrm{anc}}$ captures how close all such tokens are to a common target $V^*(Q)$.

### Condition 2: Anchor emission probability

*Informally*: The reasoning policy emits anchor tokens with some baseline probability at each step.

*Formally*: There is a constant $p_0 > 0$ such that, conditional on the trajectory history $\mathcal{F}_{j-1}$, the policy emits an anchor token at step $j$ with probability $\ge p_0$.

*Why this is reasonable*: A reasoning model trained on math problems does not emit complete gibberish; it emits tokens correlated with the answer's region of vocab. The constant $p_0$ is a *lower bound* on how often that happens. A practitioner can probe this on a held-out set by counting anchor-region emissions per step.

### Condition 3: Anchor score margin

*Informally*: When the model attends from `</think>` to the prefix, it scores anchor tokens higher than non-anchor tokens.

*Formally*: There is a constant $\Delta > 0$ such that, uniformly along the trajectory and across $Q$, the softmax pre-activation score for any anchor key minus the score for any non-anchor key is at least $\Delta$.

*Why this is reasonable*: This is a property of the trained attention pattern. If the model has learned to attend to answer-relevant tokens, anchor tokens should systematically score higher than non-anchor tokens. The margin $\Delta$ controls how much softmax mass eventually lands on anchors.

### Condition 4: Bounded value norms

*Informally*: Value vectors are bounded.

*Formally*: $\|V_k\| \le M$ for all keys $k$.

*Why this is reasonable*: Standard sanity. Layer normalization in modern transformers guarantees this.

### Condition 5: Decoding margin

*Informally*: If the attention output is close enough to $V^*(Q)$, the decoded answer is correct.

*Formally*: If $\|x - V^*(Q)\| \le \gamma(Q)$, then $\mathrm{decode}(x)$ produces a correct answer for $Q$. The decoding margin $\gamma(Q)$ may depend on the question.

*Why this is reasonable*: Trained models have margin in their final-token logits. The exact value of $\gamma(Q)$ depends on how separated the correct answer's logit is from competitors at decoding. We treat decoding as a black box and require this margin only.

Conditions 1–3 are about the trained model's inference behavior, and all three are measurable from forward passes (no peeking at training data). Conditions 4–5 are sanity. None of the five requires opening the model's training distribution.

## The result

With these five conditions, plus a relationship between $\Delta$ and the other constants — specifically $\Delta \ge \log(4(M + \max_Q \|V^*(Q)\|)/(p_0\, \gamma_{\min}))$ — here is what we get.

In LaTeX form: under the five conditions and the $\Delta$ condition, for every question $Q$ and every reasoning horizon $T \ge 1$,

$$\Pr\Big[\, \|x_T - V^*(Q)\| \;\le\; \tfrac{\gamma(Q)}{2} + \varepsilon_{\mathrm{anc}} \;\;\text{and}\;\; \mathrm{decode}(x_T) \in \mathrm{Correct}(Q) \,\Big] \;\ge\; 1 - 2\exp\!\Big(-\frac{p_0\, T}{8}\Big).$$

The failure probability decays exponentially in $T$ at rate $p_0 / 8$. As $T \to \infty$, the probability that $x_T$ lands outside the decoding-margin ball vanishes.

A two-line tail-to-expectation conversion gives the expected-error corollary:

$$\mathbb{E}\bigl[\|x_T - V^*(Q)\|\bigr] \;\le\; \tfrac{\gamma(Q)}{2} + \varepsilon_{\mathrm{anc}} \;+\; 2\bigl(M + \|V^*(Q)\|\bigr) \exp\!\Big(-\frac{p_0\, T}{8}\Big).$$

This is the test-time scaling form. There is a $T$-independent floor — $\gamma(Q)/2 + \varepsilon_{\mathrm{anc}}$, set by the trained-model constants — plus a term that decays exponentially in the reasoning horizon. Doubling thinking compute roughly squares the failure-probability bound; quadrupling roughly cubes it.

## Where the rate comes from

The proof of the theorem has three pieces. I will walk through each.

### Piece 1: How many anchors did we emit?

Define the indicator $X_j = \mathbf{1}\{a_j \in \mathcal{A}(Q)\}$ — does the $j$-th emitted token belong to the anchor set? Then the number of anchor emissions in $T$ steps is

$$\bigl|\mathcal{A}^{\mathrm{traj}}_T\bigr| \;=\; \sum_{j=1}^T X_j.$$

By Condition 2, each $X_j$ has conditional mean $p_j \ge p_0$, so on average we expect at least $p_0 T$ anchors in $T$ steps. But this is in expectation; the actual realization can fluctuate. We need a lower bound that holds with high probability.

Two natural tools apply: **Azuma–Hoeffding** treats $X_j - p_j$ as a generic bounded-difference martingale, while **multiplicative Chernoff** (or its martingale generalization, Freedman's Bernstein bound) exploits the fact that $X_j$ is Bernoulli — its conditional variance $p_j(1 - p_j) \le p_j$ is much smaller than the bounded-difference proxy 1. The two give exponents that differ by a factor of $1/p_0$, so for empirically realistic $p_0 \in [0.05, 0.2]$ Chernoff is $5$–$20\times$ sharper. We'll use Chernoff.

The bound we need (for sums of conditionally Bernoulli variables adapted to a filtration, at relative deviation $\delta = 1/2$):

$$\Pr\!\Bigl[\sum_{j=1}^T X_j \le \tfrac{1}{2} \sum_{j=1}^T p_j\Bigr] \;\le\; \exp\!\Bigl(-\tfrac{1}{8} \sum_{j=1}^T p_j\Bigr).$$

The exponent $1/8$ is the value of $\delta^2 / 2$ at $\delta = 1/2$. (The reason it's not $\delta^2/2 \cdot \mu = \mu/8$ verbatim: multiplicative Chernoff has the form $\exp(-\delta^2 \mu / (2 + \delta))$ at general $\delta$; the constant we wrote is the clean $\delta = 1/2$ case.) The martingale generalization is in Freedman 1975, Theorem 1.6.

Now substitute $\sum_j p_j \ge p_0 T$ (from Condition 2):

$$\Pr\!\Bigl[\sum_{j=1}^T X_j \le \tfrac{p_0 T}{2}\Bigr] \;\le\; \exp\!\Bigl(-\frac{p_0 T}{8}\Bigr).$$

Call the good event $\mathcal{E}_1 := \{|\mathcal{A}^{\mathrm{traj}}_T| \ge p_0 T / 2\}$. We have $\Pr[\mathcal{E}_1] \ge 1 - \exp(-p_0 T/8)$, and on $\mathcal{E}_1$, $|\mathcal{A}^{\mathrm{traj}}_T| \ge p_0 T / 2$.

The exponential rate in $T$ comes from this step. The rest of the proof is deterministic given $\mathcal{E}_1$.

### Piece 2: How much softmax mass do anchors carry?

On $\mathcal{E}_1$, at least $p_0 T / 2$ of the $T$ emitted tokens are anchors. The softmax weight on each anchor position is

$$w_{T,k} \;=\; \frac{\exp(q^\top k_k)}{s_T}, \qquad k \in \mathcal{A}^{\mathrm{traj}}_T.$$

We want to lower-bound the *total* anchor mass $W_{\mathcal{A}}(T) := \sum_{k \in \mathcal{A}^{\mathrm{traj}}_T} w_{T,k}$, equivalently upper-bound the non-anchor mass $1 - W_{\mathcal{A}}(T)$.

Condition 3 says any anchor score $\sigma_a$ exceeds any non-anchor score $\sigma_n$ by at least $\Delta$. Using $\exp(\sigma_n) \le \exp(\sigma_a - \Delta) = e^{-\Delta} \exp(\sigma_a)$,

$$1 - W_{\mathcal{A}}(T) \;=\; \frac{\sum_{k \notin \mathcal{A}^{\mathrm{traj}}_T} \exp(\sigma_k)}{\sum_k \exp(\sigma_k)} \;\le\; \frac{|\mathrm{non}\text{-}\mathrm{anchors}| \cdot \max_n \exp(\sigma_n)}{|\mathrm{anchors}| \cdot \min_a \exp(\sigma_a)} \;\le\; \frac{|\mathrm{non}\text{-}\mathrm{anchors}|}{|\mathrm{anchors}|} \cdot e^{-\Delta}.$$

On $\mathcal{E}_1$, $|\mathrm{anchors}| \ge p_0 T / 2$ and $|\mathrm{non}\text{-}\mathrm{anchors}| \le T$, so

$$1 - W_{\mathcal{A}}(T) \;\le\; \frac{T}{p_0 T / 2} \cdot e^{-\Delta} \;=\; \frac{2}{p_0}\, e^{-\Delta}.$$

So on $\mathcal{E}_1$, the non-anchor softmax mass is at most $(2/p_0) e^{-\Delta}$, independent of $T$.

### Piece 3: Triangle decomposition

We now decompose $\|x_T - V^*(Q)\|$ into the anchor contribution and the non-anchor contribution. Using $\sum_k w_{T,k} = 1$, we can write $V^*(Q) = \sum_k w_{T,k} V^*(Q)$, and then

$$x_T - V^*(Q) \;=\; \sum_{k \in \mathcal{A}^{\mathrm{traj}}_T} w_{T,k} (V_k - V^*(Q)) \;+\; \sum_{k \notin \mathcal{A}^{\mathrm{traj}}_T} w_{T,k} (V_k - V^*(Q)).$$

Take norms and apply the triangle inequality:

$$\|x_T - V^*(Q)\| \;\le\; \underbrace{\sum_{k \in \mathcal{A}^{\mathrm{traj}}_T} w_{T,k} \|V_k - V^*(Q)\|}_{\text{anchor error}} \;+\; \underbrace{\sum_{k \notin \mathcal{A}^{\mathrm{traj}}_T} w_{T,k} \|V_k - V^*(Q)\|}_{\text{non-anchor leakage}}.$$

By Condition 1, $\|V_k - V^*(Q)\| \le \varepsilon_{\mathrm{anc}}$ for $k$ an anchor. So the anchor error is

$$\sum_{k \in \mathcal{A}^{\mathrm{traj}}_T} w_{T,k} \|V_k - V^*(Q)\| \;\le\; \varepsilon_{\mathrm{anc}} \cdot \sum_{k \in \mathcal{A}^{\mathrm{traj}}_T} w_{T,k} \;\le\; \varepsilon_{\mathrm{anc}}.$$

By Condition 4 and the triangle inequality, $\|V_k - V^*(Q)\| \le M + \|V^*(Q)\|$. So the non-anchor leakage is at most

$$(M + \|V^*(Q)\|) \cdot \bigl(1 - W_{\mathcal{A}}(T)\bigr) \;\le\; (M + \|V^*(Q)\|) \cdot \frac{2}{p_0}\, e^{-\Delta}.$$

Putting the two pieces together, on $\mathcal{E}_1$:

$$\|x_T - V^*(Q)\| \;\le\; \varepsilon_{\mathrm{anc}} \;+\; \frac{2(M + \|V^*(Q)\|)}{p_0}\, e^{-\Delta}.$$

The second term is $T$-independent — it's the part the score-margin $\Delta$ has to handle, not the reasoning horizon. The $\Delta$ condition in the theorem statement ($\Delta \ge \log(4(M + \max_Q \|V^*(Q)\|)/(p_0\, \gamma_{\min}))$) is exactly what is needed to make this term at most $\gamma(Q)/2$. (Take logs of both sides; rearrange.)

So, on $\mathcal{E}_1$,

$$\|x_T - V^*(Q)\| \;\le\; \varepsilon_{\mathrm{anc}} \;+\; \gamma(Q)/2.$$

By Condition 5, this is inside the decoding margin, so $\mathrm{decode}(x_T)$ is correct.

### Combining the pieces with a union bound

We have shown:

- $\Pr[\mathcal{E}_1^c] \le \exp(-p_0 T / 8)$ (Piece 1).
- On $\mathcal{E}_1$, $\|x_T - V^*(Q)\| \le \gamma(Q)/2 + \varepsilon_{\mathrm{anc}}$, and $\mathrm{decode}(x_T)$ is correct (Pieces 2 + 3).

The "bad" probability is bounded by $\Pr[\mathcal{E}_1^c] \le \exp(-p_0 T/8)$. The factor of 2 in the theorem's $1 - 2\exp(-p_0 T/8)$ is union-bound slack from the precise writing of the proof. The conclusion follows.

This is the entire argument. The exponential rate $\exp(-p_0 T/8)$ is inherited directly from Azuma; the score-margin condition handles the deterministic part; the triangle decomposition stitches everything together.

## What this predicts

The exponential rate $\exp(-p_0 T / 8)$ is the test-time analogue of training-time scaling laws: more compute, exponentially lower failure probability, with the rate determined entirely by inference-time observables.

The most direct empirical prediction is for [Choi et al. 2025](https://arxiv.org/abs/2509.26522). They observe that the next-token entropy at the `</think>` position decreases and plateaus with thinking length. Where does that plateau come from? A short calculation pins it down: the next-token logits are $W_U x_T$ (unembedding applied to our running average), so the entropy of the resulting softmax distribution is a Lipschitz function of $x_T$. Pushing the $\|x_T - V^*(Q)\|$ bound through the Lipschitz composition gives an entropy-decay corollary:

$$\mathbb{E}\bigl[\,|H_T - H_\infty(Q)|\,\bigr] \;\le\; L_{\mathrm{sm}}\, B_U \cdot \bigl(\gamma(Q)/2 + \varepsilon_{\mathrm{anc}}\bigr) + 2 L_{\mathrm{sm}}\, B_U\, (M + \|V^*(Q)\|)\, \exp(-p_0 T / 8),$$

where $L_{\mathrm{sm}}$ is the Lipschitz constant of $\mathbf{x} \mapsto H(\mathrm{softmax}(\mathbf{x}))$ and $B_U$ bounds the operator norm of the unembedding matrix. The entropy converges to a model-dependent floor at the same exponential rate $\exp(-p_0 T / 8)$. The plateau Choi et al. threshold for early exit is the empirical signature of this convergence; the rate above predicts how quickly it should appear.

Two adjacent stories handle complementary pieces. The looped/latent dynamics analyzed by Saunshi and Geiping cover hidden-state convergence *inside* a recurrent block; we cover attention-output convergence *across* emitted tokens at decode time. Chen et al. and Kim et al. get exponential failure-decay from parallel sample aggregation; we get it from within-chain weighted averaging. Exponential decay is, evidently, a destination reachable by several mechanisms.

## Is the rate tight?

The bound $\Pr[\text{failure}] \le 2\exp(-p_0 T / 8)$ is an *upper* bound. Could a sharper proof give a faster decay?

The exponential dependence on $p_0 T$ is tight. Here's a construction. Take an adversarial question $Q^*$ where the model emits an anchor with conditional probability *exactly* $p_0$ at every step, independent of history (so the policy is Bernoulli-iid in its anchor emissions). The event "no anchor was emitted in $T$ steps" has probability

$$\Pr[\,|\mathcal{A}^{\mathrm{traj}}_T| = 0\,] \;=\; (1 - p_0)^T.$$

On this event, the running average has zero softmax mass on any anchor token; non-anchors carry all the weight, and the running average sits at distance $\ge \gamma(Q^*)$ from $V^*(Q^*)$ — outside the decoding margin. So the decoded answer is wrong:

$$\Pr[\,\mathrm{decode}(x_T) \notin \mathrm{Correct}(Q^*)\,] \;\ge\; (1 - p_0)^T \;\ge\; \exp\bigl(-T \log(1/(1-p_0))\bigr).$$

For small $p_0$, $\log(1/(1-p_0)) \approx p_0$, so the lower bound is $\approx \exp(-p_0 T)$. The upper bound is $\exp(-p_0 T / 8)$. The exponents match up to a constant factor of $8$ — the dependence $p_0 T$ in the exponent is **the right rate**, up to constants.

## Going one step further: accuracy itself can scale

The main theorem is a *confidence* scaling result. The probability of failure decays as $T$ grows, but the *floor inside the success event*, $\gamma/2 + \varepsilon_{\mathrm{anc}}$, is $T$-independent. More thinking makes you more confident you got it right within that floor; it doesn't push the floor down.

Under a stronger assumption on anchor values, both can be achieved. Replace **Condition 1** (anchor accuracy, pointwise bound) with:

**Condition 1' (Anchor unbiasedness).** Anchor value vectors are unbiased estimators of $V^*(Q)$ with bounded variance:

$$\mathbb{E}[V(k_a) \mid a \in \mathcal{A}(Q)] \;=\; V^*(Q), \qquad \mathbb{E}\|V(k_a) - V^*(Q)\|^2 \;\le\; \sigma^2.$$

This is strictly stronger than Condition 1 — the pointwise bound is replaced by a zero-mean condition. Add a mild anchor-internal uniformity hypothesis (the softmax score gap *among* anchor tokens is bounded by some $\Delta'$, so that no single anchor monopolizes the softmax mass on anchors) and the variance-of-averaged-estimator argument kicks in. Averaging $\Omega(p_0 T)$ unbiased anchor value vectors gives an estimator whose squared error scales as $1/T$:

$$\mathbb{E}\bigl[\|x_T - V^*(Q)\|^2\bigr] \;\le\; \frac{4 e^{\Delta'} \sigma^2}{p_0\, T} + 2\bigl(M + \|V^*(Q)\|\bigr)^2 \exp\!\Bigl(-\frac{p_0 T}{8}\Bigr).$$

Taking square root by Jensen,

$$\mathbb{E}\|x_T - V^*(Q)\| \;\le\; \frac{2 e^{\Delta'/2} \sigma}{\sqrt{p_0 T}} \;+\; \sqrt{2}\,(M + \|V^*(Q)\|)\, \exp\!\Bigl(-\frac{p_0 T}{16}\Bigr).$$

The floor itself decays as $1/\sqrt{T}$ now. This is the difference between *"more thinking makes me more confident I got it right"* and *"more thinking makes my answer actually better."* Confidence scaling, vs. accuracy scaling.

The unbiasedness hypothesis is more demanding than the uniform pointwise bound of Condition 1. Empirically, it would correspond to the case where the model's anchor tokens encode different but mutually-consistent approximations of $V^*(Q)$ — paraphrases of the same answer, alternative intermediate-step framings, equivalent but lexically-distinct phrasings. Whether this holds in practice for real reasoning models is an empirical question worth measuring directly.

## When does this framework apply, and when does it break?

The rate $\exp(-p_0 T / 8)$ is parameter-free in $T$ — it depends on three model-side observables ($p_0$, $\Delta$, $\varepsilon_{\mathrm{anc}}$) and one question-side constant ($\gamma(Q)$). The substantive empirical question is for which (model, problem-field) pairs the five conditions actually hold.

**Tasks the framework targets.** I designed it with verifiable-reward reasoning in mind: mathematics, code synthesis, discrete logical-deduction problems. For these the anchor set $\mathcal{A}(Q)$ has a natural realization — answer digits, the contents of a `\boxed{}`, tokens after an "Answer:" marker, or synonyms thereof. Condition 3 (the score margin $\Delta$) is the inference-time signature of an attention head trained to attend to answer-bearing positions, which is the empirically dominant behavior reported for reasoning-tuned models.

**Tasks it does not cover.** Two regimes break the assumptions before any constants are estimated. *Open-ended generation* — creative writing, summarization, free-form dialogue — has no discrete correct set with a margin around it; Condition 5 has no natural instantiation. *Long-horizon planning and agentic tasks* localize progress not at the token level but across tool calls and subgoals; the value-projection signature need not concentrate around a single $V^*(Q)$. The framework can be applied per-subgoal under a hierarchical decomposition, but I don't formalize that here.

**Where the model-side constants come from.** $p_0$ is a property of the trained policy: larger reasoning models and models post-trained with verifiable rewards re-mention answer-bearing tokens more frequently, giving higher $p_0$. $\Delta$ is a property of the trained attention pattern at the layer-head pair; sharper attention to anchors (the post-RL signature) gives larger $\Delta$. $\gamma(Q)$ is question-dependent: easy questions have wide decoding basins; near-threshold questions have $\gamma(Q) \to 2\varepsilon_{\mathrm{anc}}$ and the bound barely guarantees anything. The framework predicts test-time scaling should be most visible on intermediate-difficulty questions.

**Three falsifiable predictions** (no retraining required):

1. **Pass@1 vs. $T$ shape.** Estimate $p_0$ from anchor-emission counts and $\Delta$ from post-softmax weight readout on a held-out trajectory subset. Plug into $\exp(-p_0 T / 8)$. The prediction is the *qualitative shape* — exponential decay at rate $p_0/8$ with a question-dependent floor — not the absolute constant in front.

2. **Entropy plateau rate.** Choi et al. 2025 observe the entropy of next-token softmax at `</think>` plateauing as $T$ grows. The entropy-decay corollary predicts the plateau is approached at rate $\exp(-p_0 T / 8)$. Measure both; they should agree up to a constant.

3. **Per-question heterogeneity.** Two questions $Q_1, Q_2$ with comparable difficulty but different anchor densities (a single-digit answer vs. a multi-token expression) should show parallel exponential decays (same rate $p_0$) with different asymptotes (different $\gamma(Q_i)/2 + \varepsilon_{\mathrm{anc}}$ floors).

**Three honest caveats.** The five conditions are sufficient, not necessary — failure of one doesn't certify the scaling fails, only that this proof doesn't certify it. The theorem bounds the *probability* of landing in the decoding margin; it says nothing about whether the reasoning trace is "faithful" to the answer it produces (a model that post-hoc rationalizes can still satisfy the theorem). The policy is treated as fixed; in practice RL training itself improves $p_0$ over time. A full account of test-time scaling needs both this within-inference analysis and a training-time analysis of how $p_0$ grows.

## Future directions

The recurrence $x_j = (1 - \lambda_j) x_{j-1} + g_j$ has more structure than the proof uses. Four directions feel worth pursuing.

**Optimization-style diagnostics for reasoning models.** $p_0$, $\Delta$, $\varepsilon_{\mathrm{anc}}$ are inference-time observables on any open-weights model. Reporting them alongside pass@1 gives a structural breakdown of "how fast does this model think" — comparable across families the way loss curves and gradient norms are comparable across training runs. I expect these constants to be more diagnostic than pass@1 alone, since pass@1 conflates the rate $p_0$ and the floor.

**Momentum-style acceleration of thinking.** The recurrence has SGD form; standard acceleration (Polyak / Nesterov momentum, adaptive step sizes, second-moment scaling à la Adam) operates by reshaping the effective weight-decay schedule. Applied at inference as a temperature schedule that reshapes $\lambda_j$ over $j$, can the same techniques give a strictly faster within-inference rate? Open questions: does reshaping preserve the score margin? Is the inference-time overhead paid back by the reduction in $T$? This is the most directly actionable direction and could be tested without retraining.

**Sharpness-aware reasoning.** The floor $\gamma/2 + \varepsilon_{\mathrm{anc}}$ depends on where in the decoding-margin ball $x_T$ lands. SAM-style perturbation of intermediate latents $x_j$ (perturb in a small ball, take worst case, continue) might trade extra forward passes for a tighter effective floor. The cost is multiplicative in compute; the payoff is unclear in this formalism but worth testing on generalization-sensitive benchmarks.

**Compounding within-chain and across-sample scaling.** The bound $\exp(-p_0 T / 8)$ governs failure probability *within* a single reasoning chain. A complementary line of work considers drawing $N$ parallel chains and aggregating (majority vote, best-of-$N$ verifier scoring); failure probability then decays in $N$. Combining the two regimes, a compute budget of $N T$ forward passes can be allocated to trade $T$ (longer thinking) against $N$ (more samples). A compound bound $\exp(-p_0 T / 8 - c\, N)$ would formalize the trade-off and predict the optimal $(N, T)$ split as a function of model constants. I see this as the most useful extension for deploying reasoning models at fixed inference budgets.

## Coda

The proof above — three theorems, two corollaries, seven lemmas, all the assumptions, the citation digests, the confidence trace, four rounds of self-review — was drafted end-to-end by an Agent Skill called [`dlt-proof-writing`](https://github.com/ChristianYang37/DLT-Proof-Writing-Skill/tree/main). What I spent my time on was the part this post does — thinking through the idea, and judging whether the structure and assumptions Claude Code proposed were reasonable; the bookkeeping side ran on the skill's own pipeline. The mechanics of the skill and what it automates are covered in my next post: [Introducing dlt-proof-writing]({{< ref "posts/02-introducing-dlt-proof-writing" >}}).

Full PDF (with all constants, lemma proofs, and the three formal theorem statements): [01-reasoning-as-optimization.pdf](https://github.com/ChristianYang37/DLT-Proof-Writing-Skill/blob/main/eval_results/08-reasoning-as-optimization/pdf/main.pdf).
