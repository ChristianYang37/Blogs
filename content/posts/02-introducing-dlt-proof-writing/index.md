---
title: "dlt-proof-writing: An Agent Skill for LaTeX Proofs in Deep Learning Theory"
date: 2026-05-25T02:00:00+08:00
math: true
---


`dlt-proof-writing` is an Agent Skill (for Claude Code, or any Anthropic-Agent-Skills-compatible runtime) that automates the bookkeeping side of writing a proof in deep-learning theory. Writing a proof splits roughly in two: finding the structure and picking the assumptions on one side; drawing the dependency graph, looking up citations, getting the format consistent, running lint, doing the read-through-twice-and-fix pass on the other. The skill takes the second half. I keep the first.

It targets proofs in optimization, learning theory, statistical rates, fine-grained complexity, and RL regret.

Open source under CC BY-NC 4.0. Source: [github.com/ChristianYang37/DLT-Proof-Writing-Skill](https://github.com/ChristianYang37/DLT-Proof-Writing-Skill).

## A worked example: test-time scaling for thinking LLMs

The companion post, [Reasoning as Optimization]({{< ref "posts/01-reasoning-as-optimization" >}}), works through a specific result I produced using this skill. The question is a piece of an active research conversation: when an o1- or R1-style LLM spends more compute on "thinking", what is the *mechanistic* rate at which accuracy improves?

I drafted the proof end-to-end with `dlt-proof-writing`. The output is a 20-page PDF carrying three theorems, two corollaries, seven lemmas, and a discussion section on empirical scope. Here's the page with the main result:

![Main theorem and Corollary 10.2 in the compiled PDF](01-proof-page-10-main-theorem-corollary.png)

Under five inference-time-observable assumptions on the LLM's reasoning policy, the failure probability of the thinking process decays exponentially in the reasoning horizon $T$ at a rate determined by the anchor-emission probability $p_0$:

$$\Pr[\text{failure}] \;\le\; 2 \exp\!\Bigl(-\frac{p_0\, T}{8}\Bigr).$$

A two-line tail-to-expectation argument gives the expected-error corollary:

$$\mathbb{E}\bigl[\|x_T - V^*(Q)\|\bigr] \;\le\; \tfrac{\gamma(Q)}{2} + \varepsilon_{\mathrm{anc}} + 2(M + \|V^*(Q)\|)\,\exp\!\Bigl(-\frac{p_0\, T}{8}\Bigr).$$

The proof goes further. A second corollary translates the running-average convergence into next-token-entropy decay (matching the plateau Choi et al. 2025 document empirically); a separate theorem in a "discussion" section gives a *lower bound* showing the $\exp(-p_0 T)$ rate is tight up to a constant in the exponent; and a third theorem in another section shows that under a stronger unbiasedness hypothesis on anchor values, the floor itself decays as $\sigma/\sqrt{p_0 T}$ — i.e., accuracy scales with $T$, not just confidence:

![Variance-reduced accuracy-scaling theorem](01-proof-page-15-variance-reduced.png)

All `.tex` sources, the dependency graph, citation digests, technique digests, the confidence trace, and five iterations of self-review live in [`eval_results/08-reasoning-as-optimization/`](https://github.com/ChristianYang37/DLT-Proof-Writing-Skill/tree/main/eval_results/08-reasoning-as-optimization). The compiled PDF is [here](https://github.com/ChristianYang37/DLT-Proof-Writing-Skill/blob/main/eval_results/08-reasoning-as-optimization/pdf/main.pdf) (20 pages).

What this gets me: the structure-finding and assumption-picking happen in conversation; everything else — bibliography, lemma scaffolding, lint, five rounds of self-review — happens by the skill's own pipeline. Total time I spend on bookkeeping is effectively zero.

## The four-phase workflow

The skill enforces a sequence the agent cannot skip:

1. **Phase A — Plan.** Read the project, declare scope (Quick / Standard / Appendix) and have `check_scope.py` verify the declaration is not understated, enumerate the non-trivial techniques the proof will need, write a `.proof-research/<technique>.md` digest for each, pick organizational patterns from a catalogue, decompose the goal into a dependency graph of named lemmas. Any lemma without a downstream consumer gets deleted before drafting starts. There is no "we'll use it later" exception.

2. **Phase B — Preliminaries.** Notation, definitions, assumptions. Each assumption is followed by exactly one `\begin{remark}` discussing its mildness and weakenability. The discussion goes into the remark; the assumption stays clean.

3. **Phase C — Statements and proofs.** Topological order, leaves first. Per-statement review (a seven-item checklist). Per-proof review (a nine-item checklist; the third item is the silent-bug favourite — does the cited lemma's hypothesis actually hold at the cite-site?). One section per `.tex` file. Incremental lint after every write.

4. **Phase C.5 — Confidence sweep.** I tag every derivation step 🔴 `from-memory` by default. The author agent walks the flat list and upgrades each step via a fast path (textbook inequality / digest match / project-lemma match) or by firing a verifier sub-agent that re-derives the step in the background. Whatever is still 🔴 at the end carries a `\todo{verify}` marker so the Phase D reviewer focuses on the right places.

5. **Phase D — End-to-end review.** Three scripted gates each have to exit 0 (compile, lint, confidence coverage). A bounded peer-review loop spawns a reviewer sub-agent; each weakness gets classified as REAL-blocking / REAL-nonblocking / PHANTOM / INTENTIONAL; minimum-change fixes apply; the loop converges or hits a 3-round cap.

Each phase boundary is gated by an exit code or a script verdict, not by the agent's own judgement.

## Three design choices

**The triple Phase-D gate.** Three checks I would do manually before showing a draft to a collaborator, run automatically. `latexmk-wrapper.py` ensures the document compiles. `lint.py` enforces formatting and citation hygiene. `check_confidence_tags.py` confirms every derivation step has been annotated with its confidence level, with at least 50% trace coverage. Any one of them returning non-zero blocks the review loop. The hour of mechanical re-reading I would otherwise spend before sending out a draft is what gets automated here.

**Mechanical honesty triggers.** Every time I've written a proof and gone back to check citations and constants, I've found at least one place where the cited lemma's hypothesis didn't quite hold at the cite-site, or where I introduced a constant without saying what it depended on. The lint rules check those automatically:

- Writing `\cite{key}` without a matching `.proof-research/cite-<key>-*.md` digest fires R13.
- Invoking matrix Bernstein or Hanson-Wright without `.proof-research/<technique>.md` fires R14.
- Introducing a bare `$C$` without the universal-constant declaration fires R15.
- A case-split that says "the other case is similar" when the cases use different machinery fires R16.
- Stating a `1-\delta` conclusion without an explicit union-bound paragraph fires R17.
- Putting any theorem or proof environment inside `main.tex` itself fires R18.

These are pre-conditions to "I'm done". The bookkeeping bugs that used to take a careful re-read to find now block the workflow at lint.

**Confidence sweep with fire-and-forget verifier sub-agents.** When I write a proof by hand, I carry a rough map of which steps I am certain about and which I would re-derive before submitting. The confidence sweep externalises that map. Every derivation step starts tagged 🔴 by default, and the agent upgrades each step either by matching to a digest in the technique catalogue or by spawning a verifier sub-agent that re-derives the step independently. The author keeps writing; verifiers run in the background. Whatever survives 🔴 through the sweep gets surfaced to the Phase D reviewer with a pointer.

## Eval coverage

I validated the skill on seven calibration benchmarks: five in-scope DLT proofs (Hoeffding's inequality, NTK two-layer convergence, VC generalization bound, LSVI-UCB regret on a Linear MDP, Sobolev minimax lower bound) and two out-of-DLT generalization probes (the Ellenberg–Gijswijt cap-set bound, the Gilmer union-closed bound via entropy). All 70 assertions pass under the full workflow.

The two probes are where the signal lives. Slice rank and entropy methods sit in a different mathematical neighbourhood from NTK and Polyak-Łojasiewicz, and the gates apply the same way. The discipline carries.

## Getting started

Plugin install:

```
/plugin marketplace add ChristianYang37/DLT-Proof-Writing-Skill
/plugin install dlt-proof-writing@DLT-Proof-Writing-Skill
```

Manual install (symlink; edits to the source repo take effect immediately):

```
git clone https://github.com/ChristianYang37/DLT-Proof-Writing-Skill.git
cd DLT-Proof-Writing-Skill
make install
```

A reusable XML-tagged prompt template lives in the [README](https://github.com/ChristianYang37/DLT-Proof-Writing-Skill#-reusable-prompt-template). Fill `<problem>` with the formal statement; leave `<approach>`, `<proof_structure>`, `<target_theorems>` blank when you want the skill to propose them. The blank-handling protocol stops the agent at the end of Phase A so you sign off on the approach before any LaTeX gets drafted.

Cite as:

```bibtex
@misc{dlt-proof-writing-skill,
  author       = {Yang, Chiwun},
  title        = {{DLT} {P}roof {W}riting {S}kill: an {A}gent {S}kill for rigorous deep-learning-theory proof drafting in {L}a{T}e{X}},
  year         = {2026},
  howpublished = {GitHub: \url{https://github.com/ChristianYang37/DLT-Proof-Writing-Skill}},
  note         = {Licensed under CC BY-NC 4.0}
}
```

---

← Back to [Blog 1: Reasoning as Optimization]({{< ref "posts/01-reasoning-as-optimization" >}})
