# NMask2 外生时间信息分组门控

对应实现参数：`"channel_group_gating": true`。默认值为 `false`；关闭时保持原有的局部 patch 与 Exo-Summaries 联合 softmax。

## 1. 符号定义

- $h_p \in \mathbb{R}^{D}$：第 $p$ 个目标 patch 的表示。
- $z_{j,p} \in \mathbb{R}^{D}$：第 $j$ 个外生变量在第 $p$ 个 patch 的表示。
- $S_j=\{s_{j,1},\ldots,s_{j,R}\}$：第 $j$ 个外生变量的 $R$ 个 Exo-Summaries。
- $W$：历史局部窗口长度，包含当前位置，因此严格历史部分包含至多 $W-1$ 个 patch。
- $q_p=W_Qh_p$：目标 patch 的 query。

## 2. 三类外生表示

当前位置表示为

$$
c^{\mathrm{cur}}_{p,j}=W_Vz_{j,p}.
$$

历史窗口只包含当前位置之前的外生 patch：

$$
\mathcal{H}_{p,j}=\{z_{j,p-W+1},\ldots,z_{j,p-1}\}.
$$

历史表示为

$$
c^{\mathrm{hist}}_{p,j}
=
\operatorname{Attn}\!\left(
q_p,
K(\mathcal{H}_{p,j}),
V(\mathcal{H}_{p,j})
\right).
$$

摘要表示为

$$
c^{\mathrm{sum}}_{p,j}
=
\operatorname{Attn}\!\left(
q_p,
K(S_j),
V(S_j)
\right).
$$

历史窗口和 Exo-Summaries 分别进行注意力归一化，不再与当前位置共用一个 softmax。

## 3. 动态门控

使用在所有目标位置和外生变量之间共享的门控网络：

$$
g^{\mathrm{hist}}_{p,j}
=
\sigma\!\left(
w_h^{\top}
[h_p;c^{\mathrm{cur}}_{p,j};c^{\mathrm{hist}}_{p,j}]
+b_h
\right),
$$

$$
g^{\mathrm{sum}}_{p,j}
=
\sigma\!\left(
w_s^{\top}
[h_p;c^{\mathrm{cur}}_{p,j};c^{\mathrm{sum}}_{p,j}]
+b_s
\right).
$$

其中 $[\cdot;\cdot]$ 表示特征维拼接，两个门控均为 $[0,1]$ 范围内的标量。

## 4. 分组融合

第 $j$ 个外生变量针对目标 patch $p$ 的条件表示为

$$
\boxed{
c_{p,j}
=
c^{\mathrm{cur}}_{p,j}
+g^{\mathrm{hist}}_{p,j}c^{\mathrm{hist}}_{p,j}
+g^{\mathrm{sum}}_{p,j}c^{\mathrm{sum}}_{p,j}
}
$$

随后保持现有的跨外生变量多头交叉注意力：

$$
\Delta h_p
=
\operatorname{CrossAttn}\!\left(
h_p,
\{c_{p,1},\ldots,c_{p,C}\}
\right).
$$

因此，第一阶段决定同一个外生变量中“当前位置、历史窗口、摘要”的贡献，第二阶段决定不同外生变量的贡献。

## 5. 初始化

建议设置

$$
w_h\approx 0,\qquad w_s\approx 0,
\qquad b_h=b_s=-3.
$$

此时初始门控约为

$$
\sigma(-3)\approx 0.047.
$$

模型训练初期主要使用同位置的已知未来外生信息，历史窗口和 Exo-Summaries 作为较弱的残差补充；训练后可以根据样本内容提高或降低这两部分的权重。

当 $W=1$ 时不存在严格历史 patch，可令 $c^{\mathrm{hist}}_{p,j}=0$、$g^{\mathrm{hist}}_{p,j}=0$。当 $R=0$ 时同理关闭摘要分支。
