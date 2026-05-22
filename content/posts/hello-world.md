---
title: "Hello World — LaTeX Rendering Test"
date: 2026-05-22
draft: false
math: true
tags:
  - meta
  - latex
categories:
  - misc
summary: "First post — verifies KaTeX rendering of inline math, display math, matrices, and aligned equations."
---

This is the first post on the blog. Its purpose is to verify that the **LaTeX rendering pipeline** (KaTeX) and full-text **search index** are both working end to end.

## Inline math

Energy–mass equivalence: $E = mc^2$. The golden ratio is $\varphi = \dfrac{1+\sqrt{5}}{2} \approx 1.618$.

## Display math

The Gaussian integral:

$$
\int_{-\infty}^{\infty} e^{-x^2}\, dx = \sqrt{\pi}
$$

The softmax function used in classification heads:

$$
\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}, \quad i = 1, \dots, K
$$

## Aligned equations

The cross-entropy loss derivation:

$$
\begin{aligned}
\mathcal{L}(\theta) &= -\sum_{i=1}^{N} \sum_{k=1}^{K} y_{i,k} \log p_{i,k}(\theta) \\
&= -\mathbb{E}_{(x,y) \sim \mathcal{D}}\left[ \log p_\theta(y \mid x) \right] + O(1/N)
\end{aligned}
$$

## Matrices

$$
A = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
$$

## A theorem-style block

> **Theorem (Universal approximation, informal).** Let $\sigma$ be a non-polynomial continuous activation. For any continuous $f : [0,1]^d \to \mathbb{R}$ and any $\varepsilon > 0$, there exists a two-layer neural network $\hat{f}$ such that $\|\hat{f} - f\|_\infty < \varepsilon$.

---

If everything above renders as proper mathematical typography (not raw `$...$` strings), the LaTeX pipeline is working. Try the search page next.
