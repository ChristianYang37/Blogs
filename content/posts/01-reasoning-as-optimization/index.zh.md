---
title: "把推理过程当作优化：给 test-time scaling 一个具体速率"
date: 2026-05-25T02:00:00+08:00
math: true
---


推理 LLM 在 `<think>` 和 `</think>` 之间多花算力、准确率随之上升——这件事从 OpenAI o1 和 DeepSeek R1 起家时还是发布会的核心卖点，到现在 GPT-5.5、Opus 4.7、Gemini 3.5、Qwen 3.6、Kimi K2.5 等等都默认带 thinking 模式，曲线已经从「卖点」变成「基本面」：没人再单独宣传它了，但每条产品线都还在跟着这条曲线走 ([Snell 2024](https://arxiv.org/abs/2408.03314); [OpenAI 2024](https://arxiv.org/abs/2412.16720); [DeepSeek 2025](https://arxiv.org/abs/2501.12948); [Muennighoff 2025](https://arxiv.org/abs/2501.19393))。问题反而更尖锐了——**这个速率到底是多少、由什么决定**？

本文我们从一个具体的角度切入：固定 transformer 某一层、某一个注意力头，看 `</think>` 这个位置上的注意力输出随推理轨迹增长怎么演化。结论是：这个注意力输出在结构上就是一个**对所有已生成 token 的值向量做的 softmax 加权平均**。在对模型的一组温和条件下，它**指数级地收敛到正确答案对应向量的小邻域**——具体地，对任意推理步数 $T \ge 1$：

$$\Pr[\text{fail}] \;\le\; 2\exp\!\Big(-\frac{p_0\, T}{8}\Big),$$

其中 $p_0$ 是「答案相关 token 在每一步被生成的概率」的下界（下文细说）。

下面我们一步一步推。

## 一、先看看已有的视角

文献里的回答主要有几条线。

**思维链 ≈ 多步梯度下降**。[von Oswald 等 2022](https://arxiv.org/abs/2212.07677) 和 [Akyürek 等 2022](https://arxiv.org/abs/2211.15661) 证明线性自注意力加合适的权重，就能实现一步对上下文回归任务的梯度下降；近期 [Cheng 等 ICLR 2025](https://arxiv.org/abs/2502.21212) 进一步证明，思维链让单层 transformer 从一步回归升级到可证明的多步优化。这是「推理即优化」最干净的形式化版本——但停留在合成的上下文学习任务 + 训练时。

**多样本聚合带来的缩放律**。[Wang 等 2022](https://arxiv.org/abs/2203.11171) 的 self-consistency、[Yao 等 2023](https://arxiv.org/abs/2305.10601) 的 tree-of-thoughts、[Chen 等 NeurIPS 2025](https://arxiv.org/abs/2411.19477) 都证明：在测试时算力上，失败概率指数衰减或者幂律衰减。但作用机制是 **跨多条并行采样的聚合**，不是单条思考链内部的动力学。

**循环 / 隐式递归**。[Saunshi 等 ICLR 2025](https://arxiv.org/abs/2502.17416)、[Geiping 等 2025](https://arxiv.org/abs/2502.05171) 把思维链等价于对同一个 transformer 块的反复应用，观察到隐状态的收敛轨道。

**随机过程视角**。[Bondaschi 等 2026](https://arxiv.org/abs/2603.00306) 把思维链建模成隐空间状态上的马尔可夫链；[Santilli 等 2025](https://arxiv.org/abs/2506.04374) 用低维流形上的漂移-扩散方程；[Oh 2026](https://arxiv.org/abs/2601.15482) 用鞅论 + 最优停时。

经验上离我们最近的是 [Choi 等 2025](https://arxiv.org/abs/2509.26522)：他们在思考过程中持续监测 `</think>` 位置下一个 token 的熵，发现它随思考长度下降、然后进入平台，他们用这个平台的阈值做提前退出。还有一个相关结果 [Bogdan 等 2025](https://arxiv.org/abs/2506.19143)：他们发现某些专门的注意力头会从靠后的句子，把注意力广播回到前文中一小撮高重要性的 token——这正是「加权平均收敛」时应当集中到的那种结构。

我们要切入的位置是 **注意力输出这一层级**：固定 `</think>` 的位置，看 softmax 注意力在新推理 token 进入时计算什么。我要分析的对象（对所有已生成 token 值向量的 softmax 加权平均），和对应的速率（在一组锚定条件下沿推理时长的指数衰减），据我检索没有在文献里被明确写出过。

## 二、注意力输出的形式

固定 transformer 任一层 $i$、任一注意力头 $h$。在任意位置 $p$，注意力头计算

$$\mathrm{Attn}(q_p, K, V) \;=\; \sum_{k} \frac{\exp(q_p^\top k_k)}{\sum_l \exp(q_p^\top k_l)}\, V_k.$$

输出是值向量的凸组合，softmax 权重非负、和为 1。

我们要把 $p$ 选成 `</think>` 这个位置。记 $x_{i,h,T}$ 为第 $i$ 层、第 $h$ 个头、`</think>` 位置上，在生成完前 $T$ 个推理 token 之后的注意力输出。下文省略 $i, h$ 下标。

这里有个细节：`</think>` 此刻其实还没被生成。我们其实是把它当成一个**虚拟位置**——问注意力头**如果它就在那里**会算什么。它对应的查询向量，在第 $(i-1)$ 层这个位置的输出被算出来之后就有定义了，所以这个量在前向传播期间就存在，只是没被解码出来而已。

## 三、生成一个新 token 时发生什么

从 $j-1$ 个推理 token 走到 $j$ 个。Softmax 分母由

$$s_{j-1} \;=\; \sum_{k=1}^{j-1} \exp(q^\top k_k) \tag{1}$$

变成

$$s_j \;=\; s_{j-1} + \exp(q^\top k_j). \tag{2}$$

新的注意力输出是

$$x_j \;=\; \sum_{k=1}^{j} \frac{\exp(q^\top k_k)}{s_j}\, V_k. \tag{3}$$

留意到这**不是**简单地在 $x_{j-1}$ 上加一项——分母 $s_j$ 比 $s_{j-1}$ 大，所有旧权重都被缩小了一个因子 $s_{j-1}/s_j < 1$：

$$\frac{\exp(q^\top k_k)}{s_j} \;=\; \frac{s_{j-1}}{s_j} \cdot \frac{\exp(q^\top k_k)}{s_{j-1}}, \qquad k \le j-1.$$

代回 (3)：

$$x_j \;=\; \frac{s_{j-1}}{s_j} \underbrace{\sum_{k=1}^{j-1} \frac{\exp(q^\top k_k)}{s_{j-1}}\, V_k}_{=\, x_{j-1}} \;+\; \frac{\exp(q^\top k_j)}{s_j}\, V_j.$$

定义衰减因子 $\lambda_j := 1 - s_{j-1}/s_j$、增量 $g_j := (\exp(q^\top k_j)/s_j)\, V_j$，整理得

$$\boxed{\; x_j \;=\; (1 - \lambda_j)\, x_{j-1} \;+\; g_j. \;} \tag{4}$$

这就是注意力输出的「一步递推」。

## 四、题外话：这是 SGD 吗？

公式 (4) 的形式和带衰减权重的 SGD 一模一样：$\lambda_j$ 是衰减率，$g_j$ 是步长。这正是「推理是优化」这个口号的代数来源。

但稍微留意一下。SGD 里的 $g_j$ 是某个显式损失 $L$ 的随机梯度——$g_j = -\eta\, \nabla L(x_{j-1}, \text{data}_j)$。我们这里的 $g_j = \lambda_j V_j$ 是一个被重新加权的值向量，**背后并没有任何显式的 $L$**。

我们当然可以「假设」存在某个 $L$ 让 $g_j$ 是它的随机梯度，但这个 $L$ 没法从任何推理时的可观测量里读出来——它会是叠加在模型上的一个公理。下面要走的证明**不**依赖这种 $L$，只用递推 (4) 本身、把 $V_j$ 就当成模型生成的值向量。

所以：(4) 在代数上和 SGD 同形；但把它读成真的优化只是修辞，并不起承重作用。这个区别决定了我们能声称什么、能证明什么。

## 五、展开：它就是个加权平均

把 (4) 一路展开回 $x_0$。衰减因子的乘积可以折叠：

$$\prod_{i=k+1}^{j} (1 - \lambda_i) \;=\; \prod_{i=k+1}^{j} \frac{s_{i-1}}{s_i} \;=\; \frac{s_k}{s_j}.$$

所以 $g_k$ 在 $x_j$ 里的贡献是 $(s_k/s_j) \cdot g_k$。代回 $g_k = (\exp(q^\top k_k)/s_k)\, V_k$：

$$\frac{s_k}{s_j} \cdot g_k \;=\; \frac{\exp(q^\top k_k)}{s_j}\, V_k \;=:\; w_{j,k}\, V_k.$$

对 $k$ 求和：

$$\boxed{\; x_j \;=\; \sum_{k=1}^{j} w_{j,k}\, V_k, \qquad w_{j,k} \ge 0,\ \sum_k w_{j,k} = 1. \;} \tag{5}$$

事实上这就是原始的注意力公式本身。但递推 (4) 多告诉了我们一件事——**加权平均是怎么演化的**。每个新 token 把所有前面 token 的贡献按 $s_{j-1}/s_j$ 略微缩小，然后把自己加进去。

## 六、收敛到哪？

现在的动力学问题：$j \to \infty$ 时 $x_j$ 走到哪里？

如果 $V_k$ 是独立同分布、有界、期望良好，加权平均会按标准的集中不等式收敛到值向量分布的期望。但我们没有这种独立同分布结构——$V_k$ 由训练后的模型决定，给定问题 $Q$ 和已生成的 token 轨迹。

直觉要更具体一点。要让 (5) 收敛到「正确答案对应的向量」，**两件事必须发生**：

1. 词表里得有一些 token，它们的值向量已经指向正确答案附近；
2. 模型要够频繁地生成这些 token、并给它们足够高的注意力权重，让它们 dominate 整个平均。

下面我们把这两件事写成形式化的条件。

## 七、5 个条件

每个条件先用一句话讲非形式版本，再给形式版，再说为什么这样假设算合理。

**C1（锚的精度）**。对每个问题 $Q$，词表里存在一个子集 $\mathcal{A}(Q)$——叫**锚集合**——和一个目标向量 $V^*(Q)$，使得每个 $a \in \mathcal{A}(Q)$ 都满足 $\|V(k_a) - V^*(Q)\| \le \varepsilon_{\mathrm{anc}}$。

*为什么合理*：训练过的模型在做一道数学题时已经学会了"某些 token（答案的数字符号、或者"= 4"这种 pre-cursor）是 answer-bearing 的"。它们经过值投影之后的向量应当落在 $V^*(Q)$ 附近。常数 $\varepsilon_{\mathrm{anc}}$ 衡量这些 token 离公共目标 $V^*(Q)$ 的距离。

**C2（锚被生成的概率有下界）**。存在常数 $p_0 > 0$，使得给定到此为止的轨迹历史 $\mathcal{F}_{j-1}$，模型在第 $j$ 步生成锚 token 的条件概率 $\ge p_0$。

*为什么合理*：在数学上训练过的推理模型不会满嘴跑火车，它生成的 token 跟答案 region 是有相关的。$p_0$ 是这个相关频率的**下界**——拿一个 held-out set 跑几次就能估出来。

**C3（锚的注意力得分有间隙）**。存在常数 $\Delta > 0$，使得沿轨迹任意时刻、对任意问题，锚 token 的 softmax 预激活得分减去任意非锚 token 的得分 $\ge \Delta$。

*为什么合理*：这是训练后注意力模式的性质。模型学会注意 answer-relevant 的 token，那么锚就应当系统性地比非锚拿到更高的得分。$\Delta$ 控制了最终多少 softmax 质量会落到锚上。

**C4（值向量有界）**。所有 token 的 $\|V_k\| \le M$。

*为什么合理*：合理性检查。层归一化（layer normalization）直接保证。

**C5（解码间隙）**。若 $\|x - V^*(Q)\| \le \gamma(Q)$，则 $\mathrm{decode}(x)$ 输出正确答案。$\gamma(Q)$ 是问题相关的解码间隙。

*为什么合理*：训练过的模型在 final-token logit 上有 margin。$\gamma(Q)$ 是这个 margin 的量化。我们把解码当成 black box，只要求这个间隙存在。

前三条都是训练后模型推理行为的性质，三条都能从前向传播测得（不用看训练数据）。

## 八、主定理

加上一个把 $\Delta$ 和其它常数挂起来的条件 $\Delta \ge \log(4(M + \max_Q \|V^*(Q)\|)/(p_0\, \gamma_{\min}))$，结论是：

**定理（Test-time scaling for anchored attention）**。在 C1–C5 和上述 $\Delta$ 条件下，对任意 $Q \in F$ 和任意推理步数 $T \ge 1$，

$$\Pr\!\Big[\, \|x_T - V^*(Q)\| \le \tfrac{\gamma(Q)}{2} + \varepsilon_{\mathrm{anc}} \;\text{and}\; \mathrm{decode}(x_T) \in \mathrm{Correct}(Q) \,\Big] \;\ge\; 1 - 2\exp\!\Big(-\frac{p_0\, T}{8}\Big). \tag{6}$$

失败概率沿推理时长指数衰减，速率 $p_0 / 8$ 完全由锚生成概率决定。

两行从尾概率到期望的转换，给出期望误差的推论：

$$\mathbb{E}\bigl[\|x_T - V^*(Q)\|\bigr] \;\le\; \tfrac{\gamma(Q)}{2} + \varepsilon_{\mathrm{anc}} \;+\; 2(M + \|V^*(Q)\|)\,\exp\!\Big(-\frac{p_0\, T}{8}\Big). \tag{7}$$

这就是 test-time scaling 的形式——一个与 $T$ 无关的下界，加上随 $T$ 指数衰减的部分。

完整的 12 节证明（带所有常数和所有引理）在 [PDF](https://github.com/ChristianYang37/DLT-Proof-Writing-Skill/blob/main/eval_results/08-reasoning-as-optimization/pdf/main.pdf) 里；下面只走证明的主线。

## 九、证明：速率从哪里来

证明分三块。

### 第一块：T 步里生成了多少个锚？

定义指示变量 $X_j := \mathbf{1}\{a_j \in \mathcal{A}(Q)\}$，第 $j$ 步是不是锚。锚总数

$$|\mathcal{A}^{\mathrm{traj}}_T| \;=\; \sum_{j=1}^T X_j.$$

由 C2，条件期望 $p_j \ge p_0$，所以平均而言 $T$ 步里至少有 $p_0 T$ 个锚。实际实现当然会波动——我们需要一个高概率的下界。

两个工具都可以用。**Azuma–Hoeffding** 把 $X_j - p_j$ 当作一般的「步长有界的鞅差」，给出 $\exp(-p_0^2 T/8)$。**乘性 Chernoff**（它的鞅版是 Freedman 的 Bernstein 不等式）利用了 $X_j$ 是 Bernoulli 这个事实——条件方差 $p_j(1-p_j) \le p_j$ 远小于"步长有界"那种最坏情况代理 $1$，给出 $\exp(-p_0 T/8)$，**比 Azuma 紧一个 $1/p_0$ 因子**。在经验上现实的 $p_0 \in [0.05, 0.2]$ 区间，Chernoff 比 Azuma 紧 5–20 倍。我们用 Chernoff。

需要的不等式（对鞅滤波下的条件 Bernoulli 之和，相对偏差 $\delta = 1/2$）：

$$\Pr\!\Big[\sum_{j=1}^T X_j \le \tfrac{1}{2} \sum_{j=1}^T p_j\Big] \;\le\; \exp\!\Big(-\tfrac{1}{8} \sum_{j=1}^T p_j\Big). \tag{8}$$

指数里的 $1/8 = \delta^2/2$ 取 $\delta = 1/2$。鞅版推广见 Freedman 1975 Theorem 1.6。

代入 C2 给的 $\sum_j p_j \ge p_0 T$：

$$\Pr\!\Big[\sum_j X_j \le \tfrac{p_0 T}{2}\Big] \;\le\; \exp\!\Big(-\frac{p_0\, T}{8}\Big).$$

记好事件 $\mathcal{E}_1 := \{|\mathcal{A}^{\mathrm{traj}}_T| \ge p_0 T / 2\}$，则 $\Pr[\mathcal{E}_1] \ge 1 - \exp(-p_0 T/8)$，且在 $\mathcal{E}_1$ 上锚总数 $\ge p_0 T / 2$。

**指数中的 $T$ 项就是从这一步来的。** 后面两块在 $\mathcal{E}_1$ 上是确定性的。

### 第二块：锚占多少 softmax 质量？

$\mathcal{E}_1$ 上至少有 $p_0 T / 2$ 个锚。它们的 softmax 权重是 $w_{T,k} = \exp(q^\top k_k)/s_T$。

C3 给的是任意锚的得分 $\sigma_a$ 比任意非锚的得分 $\sigma_n$ 高至少 $\Delta$，即 $\exp(\sigma_n) \le e^{-\Delta} \exp(\sigma_a)$。非锚占的质量：

$$1 - W_{\mathcal{A}}(T) \;=\; \frac{\sum_{k \notin \mathcal{A}^{\mathrm{traj}}_T} \exp(\sigma_k)}{\sum_k \exp(\sigma_k)} \;\le\; \frac{|\text{non-anchor}|}{|\text{anchor}|} \cdot e^{-\Delta}.$$

$\mathcal{E}_1$ 上 $|\text{anchor}| \ge p_0 T / 2$、$|\text{non-anchor}| \le T$，所以

$$1 - W_{\mathcal{A}}(T) \;\le\; \frac{2}{p_0}\, e^{-\Delta}. \tag{9}$$

**与 $T$ 无关**——这一部分是由 $\Delta$ 控制的，不是由推理时长控制的。

### 第三块：三角不等式分解

把 $\|x_T - V^*(Q)\|$ 拆成锚的贡献 + 非锚的贡献。用 $\sum_k w_{T,k} = 1$：

$$x_T - V^*(Q) \;=\; \sum_{k \in \mathcal{A}^{\mathrm{traj}}_T} w_{T,k}\, (V_k - V^*) \;+\; \sum_{k \notin \mathcal{A}^{\mathrm{traj}}_T} w_{T,k}\, (V_k - V^*).$$

取范数、用三角不等式：

$$\|x_T - V^*\| \;\le\; \underbrace{\sum_{k \in \mathcal{A}} w_{T,k}\, \|V_k - V^*\|}_{\text{anchor error}} + \underbrace{\sum_{k \notin \mathcal{A}} w_{T,k}\, \|V_k - V^*\|}_{\text{non-anchor leakage}}.$$

由 C1，锚的误差 $\le \varepsilon_{\mathrm{anc}} \cdot W_{\mathcal{A}}(T) \le \varepsilon_{\mathrm{anc}}$。由 C4 + 三角，$\|V_k - V^*\| \le M + \|V^*\|$，所以非锚的泄漏 $\le (M + \|V^*\|) \cdot (1 - W_{\mathcal{A}}(T))$。

合并 (9)：在 $\mathcal{E}_1$ 上

$$\|x_T - V^*\| \;\le\; \varepsilon_{\mathrm{anc}} + \frac{2(M + \|V^*\|)}{p_0}\, e^{-\Delta}.$$

定理表述里 $\Delta \ge \log(4(M + \max_Q \|V^*(Q)\|)/(p_0\, \gamma_{\min}))$ 这个条件，两边取对数整理就能验证它恰好把第二项压到 $\gamma(Q)/2$ 以下。所以

$$\|x_T - V^*\| \;\le\; \varepsilon_{\mathrm{anc}} + \gamma(Q)/2 \qquad \text{on } \mathcal{E}_1.$$

由 C5，解码正确。

### 合起来

我们证了：

- $\Pr[\mathcal{E}_1^c] \le \exp(-p_0 T/8)$；
- 在 $\mathcal{E}_1$ 上 $\|x_T - V^*\| \le \gamma/2 + \varepsilon_{\mathrm{anc}}$、解码正确。

"坏"概率 $\le \exp(-p_0 T/8)$。定理里 $1 - 2\exp(-p_0 T/8)$ 的因子 2 是并集界 (union bound) 写法上留的余量。完。

整个推导其实就是三步：Azuma → Chernoff 把锚的计数变成高概率事件；C3 + softmax 算式把锚占的质量推到接近 1；三角分解把误差分两块加起来。

## 十、应用：解释熵的 plateau

指数速率 $\exp(-p_0 T/8)$ 是训练时缩放律的测试时类比——更多算力 → 失败概率指数下降。

最直接的经验对应是 Choi 等 2025 看到的熵 plateau。Plateau 在哪里来？稍微一想：下一个 token 的 logit 是 $W_U x_T$（解嵌入作用在加权平均上），所以 softmax 的熵是 $x_T$ 的 Lipschitz 函数。把 $\|x_T - V^*(Q)\|$ 的 bound 通过 Lipschitz 复合推过去，得到熵衰减的推论：

$$\mathbb{E}\bigl[|H_T - H_\infty(Q)|\bigr] \;\le\; L_{\mathrm{sm}}\, B_U \cdot \bigl(\gamma(Q)/2 + \varepsilon_{\mathrm{anc}}\bigr) + 2 L_{\mathrm{sm}}\, B_U\, (M + \|V^*(Q)\|)\, \exp(-p_0 T / 8). \tag{10}$$

其中 $L_{\mathrm{sm}}$ 是 $\mathbf{x} \mapsto H(\mathrm{softmax}(\mathbf{x}))$ 的 Lipschitz 常数、$B_U$ 是解嵌入矩阵的算子范数上界。熵沿同样的指数速率 $\exp(-p_0 T/8)$ 收敛到一个由模型决定的下界。

Choi 拿来做提前退出的那个 plateau，正是这条收敛的经验信号；公式 (10) 预测它出现得多快。

至于和别的工作的关系：Saunshi、Geiping 的循环 / 隐状态结果讲的是「循环块内部隐状态」的收敛；这里讲的是「解码时、已生成 token 上注意力输出」的收敛。Chen 等和 Kim 等用「跨并行采样聚合」拿到指数衰减；这里用「单链内部加权平均」。指数衰减看起来是几条不同路径都能到的目的地。

## 十一、速率紧吗？

(6) 是**上界**。能不能更紧？

指数中的 $p_0 T$ 是紧的。构造一下：在一个对抗性问题 $Q^*$ 上，模型每一步以**恰好** $p_0$ 概率生成锚、和历史独立（Bernoulli 独立同分布）。事件 "$T$ 步里一个锚都没生成" 的概率是

$$\Pr[|\mathcal{A}^{\mathrm{traj}}_T| = 0] \;=\; (1 - p_0)^T.$$

这个事件上，加权平均在锚 token 上的 softmax 质量为 0、整个质量跑去非锚 token，距 $V^*(Q^*)$ $\ge \gamma(Q^*)$——掉在解码间隙外，解码出错：

$$\Pr[\mathrm{decode}(x_T) \notin \mathrm{Correct}(Q^*)] \;\ge\; (1-p_0)^T \;\ge\; \exp(-T \log(1/(1-p_0))). \tag{11}$$

小 $p_0$ 时 $\log(1/(1-p_0)) \approx p_0$，下界 $\approx \exp(-p_0 T)$。上界是 $\exp(-p_0 T / 8)$。**指数 match 到常数 8**——指数里的 $p_0 T$ 就是对的速率（up to 常数）。

## 十二、再走一步：accuracy 本身也能 scale

定理 (6) 是一个**置信度**缩放结果——$T$ 涨，**失败的概率**衰减；但成功事件内部那个下界 $\gamma/2 + \varepsilon_{\mathrm{anc}}$ 与 $T$ 无关。多思考让模型更**确信**自己在那个下界内的答案对，但下界本身不动。

在锚值向量的更强假设下，两件事可以一起。把 **C1** 替换成：

**C1'（锚的无偏性）**：锚的值向量是 $V^*(Q)$ 的无偏估计、方差有界——

$$\mathbb{E}[V(k_a) \mid a \in \mathcal{A}(Q)] = V^*(Q), \qquad \mathbb{E}\|V(k_a) - V^*(Q)\|^2 \le \sigma^2.$$

这比 C1 严格强：原来的逐点界被换成「期望为零」这个条件。再加一个温和的「锚内部均匀」假设（锚 token 之间的 softmax 得分间隙由某个 $\Delta'$ 控制，避免某一个锚单独垄断锚集合上的 softmax 质量）。

然后「平均估计的方差」这个老套路就奏效了：$\Omega(p_0 T)$ 个无偏锚值向量平均起来，平方误差按 $1/T$ 衰减。具体地：

$$\mathbb{E}\bigl[\|x_T - V^*(Q)\|^2\bigr] \;\le\; \frac{4 e^{\Delta'} \sigma^2}{p_0\, T} + 2(M + \|V^*(Q)\|)^2 \exp\!\Big(-\frac{p_0 T}{8}\Big). \tag{12}$$

Jensen 开根号：

$$\mathbb{E}\|x_T - V^*(Q)\| \;\le\; \frac{2 e^{\Delta'/2} \sigma}{\sqrt{p_0 T}} \;+\; \sqrt{2}\,(M + \|V^*(Q)\|)\, \exp\!\Big(-\frac{p_0 T}{16}\Big).$$

**下界本身按 $1/\sqrt{T}$ 衰减**了。这是「多思考让我更确信我答对了」和「多思考让我的答案本身更准」的区别——前者是 confidence scaling，后者是 accuracy scaling。

无偏性比 C1 的逐点有界严格强。经验上它对应的场景是：模型的锚 token 编码了 $V^*(Q)$ 的不同但相互一致的近似——同一答案的不同表述方式、不同的中间步骤组织方式、等价但词面不同的句子。真实推理模型上这条假设到底成不成立，是一个值得直接测的经验问题。

## 十三、什么时候这个 framework 成立、什么时候不成立

速率 $\exp(-p_0 T/8)$ 在 $T$ 上无自由参数——它依赖三个模型侧的可观测量（$p_0$、$\Delta$、$\varepsilon_{\mathrm{anc}}$）和一个问题侧的常数（$\gamma(Q)$）。实质性的经验问题是：在哪些 (模型，问题领域) 组合上，5 条条件根本成立。

**框架瞄准的任务**。这个框架是为可验证奖励 (verifiable-reward) 类推理任务设计的：数学、代码合成、离散逻辑推断。在这些任务上，锚集合 $\mathcal{A}(Q)$ 有自然实现——答案的数字、`\boxed{}` 里的内容、"Answer:" 之后的 token、或者它们的同义表达。条件 3（得分间隙 $\Delta$）是 attention head 学会关注答案相关位置时的推理时签名，也是经验上 reasoning-tuned 模型的主导行为。

**框架不覆盖的任务**。有两类场景在估计常数之前就让假设失效。*开放式生成*——创意写作、摘要、自由对话——没有离散的"正确答案集合"和对应的间隙；条件 5 没法自然 instantiate。*长 horizon 规划和 agentic 任务*把进度局部化在工具调用和 subgoal 层面，不在 token 层面；值投影的特征不必集中到单一的 $V^*(Q)$。框架可以在分层分解下逐 subgoal 应用，但我们这里不形式化。

**模型侧常数从哪里来**。$p_0$ 是训练后 policy 的性质：更大的推理模型、经过可验证奖励 RL 训练后的模型，会更频繁地重复提及答案相关 token，对应更大的 $p_0$。$\Delta$ 是某一层某一头训练后注意力模式的性质；更尖锐的对锚的注意力（RL 后的典型签名）给更大的 $\Delta$。$\gamma(Q)$ 与问题相关：简单问题有宽的解码盆地；接近临界难度的问题 $\gamma(Q) \to 2\varepsilon_{\mathrm{anc}}$，bound 几乎什么都不保证。框架预测：test-time scaling 在中等难度问题上最明显。

**三个可证伪的预测**（不需要重训）：

1. **Pass@1 vs. $T$ 的形状**。从一个 held-out trajectory 子集估 $p_0$（数 anchor 生成频率）和 $\Delta$（读 softmax 后的权重）。代入 $\exp(-p_0 T/8)$。预测的是*定性形状*——速率 $p_0/8$ 的指数衰减加一个 question-dependent 的下界——而不是指数前面的绝对常数。

2. **Entropy plateau 的速率**。Choi 等 2025 观察到 `</think>` 上 next-token softmax 的 entropy 随 $T$ 增长进入 plateau。entropy-decay 推论预测这个 plateau 的逼近速率正是 $\exp(-p_0 T/8)$。两边都测，应当 up to 常数 agree。

3. **每个问题不同的下界**。两个难度相当但 anchor 密度不同的问题 $Q_1, Q_2$（比如单数字答案 vs. 多 token 的表达式），应当展现平行的指数衰减（共享 $p_0$）+ 不同的渐近线（不同的 $\gamma(Q_i)/2 + \varepsilon_{\mathrm{anc}}$ 下界）。

**三个诚实的 caveat**。5 条条件是充分非必要——一条不成立不等于 scaling 不成立，只是这个证明不再保证。定理 bound 的是落在 decoding margin 内的*概率*；它对推理 trace 是否"忠实"于产出的答案保持沉默（一个事后合理化的模型仍可能满足定理）。policy 在我们这里被当成 fixed；现实中 RL 训练本身就让 $p_0$ 增长。一个完整的 test-time scaling 账本需要同时有这套 within-inference 分析 + 一套 training-time 分析（$p_0$ 怎么涨）。

## 十四、未来方向

递推 (4) 的结构其实比证明用到的多。四个值得继续走的方向。

**用 optimization analytics 测一个推理模型的有效性**。$p_0$、$\Delta$、$\varepsilon_{\mathrm{anc}}$ 都是任何 open-weights 模型推理时的可观测量。把这些常数和 pass@1 一起报告，能给「这个模型思考得多快」一个结构性的拆解——就像 loss 曲线和梯度范数在不同训练 run 之间可比一样。我预期这些常数会比 pass@1 本身更有 diagnostic 价值，因为 pass@1 同时把速率 $p_0$ 和下界混在一起。

**用动量类方法加速思考**。递推具有 SGD 形态；经典加速技术（Polyak / Nesterov 动量、自适应步长、Adam 风格的二阶矩缩放）的本质是重塑等效的衰减率。在推理时作为一个重塑 $\lambda_j$ 的 temperature schedule 应用，能不能给一个严格更快的 within-inference 速率？开放问题：重塑会不会破坏 score margin？推理时多花的算力能不能被减少的 $T$ 换回来？这是四个方向里最直接可操作的一个，不需要重训。

**SAM 风格的 sharpness-aware 推理**。下界 $\gamma/2 + \varepsilon_{\mathrm{anc}}$ 取决于 $x_T$ 在 decoding margin 球里落到哪。SAM 风格地扰动中间 latent $x_j$（在小球内取 worst case、然后继续走），有可能拿额外的前向传播换更紧的有效下界。代价是算力上的乘法因子；收益在这套 formalism 下不明确，但值得在对泛化敏感的 benchmark 上经验测。

**Within-chain 和 across-sample scaling 复合起来**。$\exp(-p_0 T/8)$ 这个 bound 是单条推理链*内部*的。一条互补的工作线考虑抽 $N$ 条平行链做聚合（majority vote、best-of-$N$ verifier 打分）；失败概率随 $N$ 衰减。两者结合，给定 $N T$ 前向传播的算力预算，practitioner 可以在 $T$（思考更久）和 $N$（多 sample 几次）之间 trade。一个 $\exp(-p_0 T/8 - c\, N)$ 形式的复合 bound 会把这个 trade-off 形式化，并能从模型常数预测最优的 $(N, T)$ 分配。我认为这是四个方向里对在固定推理预算下部署推理模型最实用的一个。

## 收尾

上面这份证明——3 个定理、2 个推论、7 条引理、讨论章节、所有假设、引用 digest、置信度 trace、5 轮自评——是用一个叫 `dlt-proof-writing` 的 Agent Skill 端到端起草的。我自己花时间的地方是这篇 post 上面在做的事——找结构、挑假设；剩下的 bookkeeping 都跑在 skill 的 pipeline 里。Skill 的机制以及它具体自动化了什么，在配套 post 里：[Introducing dlt-proof-writing]({{< ref "posts/02-introducing-dlt-proof-writing" >}})。

完整 PDF（含所有常数、引理证明、3 个定理的形式陈述、讨论章节）：[01-reasoning-as-optimization.pdf](https://github.com/ChristianYang37/DLT-Proof-Writing-Skill/blob/main/eval_results/08-reasoning-as-optimization/pdf/main.pdf)。GitHub 源码（作为 dlt-proof-writing 的第 8 个 eval）：[`08-reasoning-as-optimization/`](https://github.com/ChristianYang37/DLT-Proof-Writing-Skill/tree/main/eval_results/08-reasoning-as-optimization)。
