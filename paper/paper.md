% [修改说明] 先提出两个问题，再依次介绍解决方法；统一使用 learnable future tokens。
% [引用说明] 保留原稿全部引用键，并按背景、交互建模、预测头设计重新安排。
% [待核对] 部分已知/部分未知的混合设置须与最终实现和实验一致。
% [复杂度说明] O(MdSP) -> O(dP) 指指定线性预测头的参数存储空间，不代表全模型空间复杂度。

% [修改 P01：背景与已有方法，补齐协变量利用相关引用]
Beyond the target (i.e., endogenous) series, exogenous variables provide complementary information for forecasting \citep{InjectTST_2025, OLIVARES2023884}. Known future covariates are particularly valuable because they describe conditions over the prediction horizon, such as holiday schedules and planned promotional campaigns. Approaches such as TimeXer \citep{Timxer_2024}, TiDE \citep{tide_2023}, and ASTNet \citep{ASTNet_2025} investigate how to incorporate exogenous information. Building on these efforts, we consider two challenges concerning the use of  future covariates.

% [修改 P02：问题一——说明现有融合视角对未来时间先验利用不足，暂不介绍 NMask]
First, most existing methods treat future exogenous variables as additional features that need to be fused into the historical sequence representation. Research has explored increasingly elaborate interactions among forecasting variables \citep{pmlr-v267-hu25ac, GinAR_2025, 10105527}. However, approaches centered on fusing future covariates into historical representations can overlook their explicit correspondence to future intervals. Future covariates specify conditions at the very timestamps for which target values must be predicted. This temporal prior motivates directly associating future target representations with the corresponding covariate observations.

% [修改 P03：问题二——部分未来协变量未知时，如何利用其演化信息]
Second, the partial availability of future covariates makes it challenging to model their future evolution and exploit it for target forecasting. Complete future information is not always accessible: calendar features may be known while future sensor measurements remain unknown. Historical observations of these covariates can still reveal information about their evolution. A model must therefore represent both observed and unobserved covariate futures and learn useful representations of the latter without accessing their values at prediction time.

% [修改 P04：解决问题一——masked prediction、未来外生时间锚点、内生未来 token 的信息聚合作用]
To address these challenges, we propose NMask, a lightweight forecasting framework based on Masked Alignment. For the first challenge, NMask reformulates forecasting with future exogenous variables as a masked prediction task, as illustrated in Figure~\ref{fig:method comparison}. Specifically, future exogenous patches are treated as temporal anchors, while a sequence of learnable tokens is appended after historical endogenous patches to represent the unknown target segments. Each token corresponds to a target variable and future patch position and is aligned with the exogenous observations for that interval. 
% Through the encoder, these tokens aggregate information from historical observations and future exogenous anchors, forming contextualized representations for predicting future endogenous patches.

% [修改 P05：共享预测头及参数空间复杂度；不将预测头优势扩大为全模型复杂度优势]
Furthermore, NMask adopts a shared predictor to reduce prediction-head parameters, complementing research on forecasting head design \citep{huang2026storm, NEURIPS2025_571c7e16, NEURIPS2025_54c9bfb0}. Each future target token predicts one patch through the same linear head. Compared with a dense head mapping $M$ historical patch embeddings to $S$ future patches, the parameter space complexity decreases from $O(MdSP)$ to $O(dP)$, where $d$ is the embedding dimension and $P$ is the patch length. This reduction concerns the predictor; encoder and token parameters are additional.

% [修改 P06：解决问题二——协变量未来 token 与辅助预测监督]
For the second challenge, NMask retains available future covariate patches and introduces learnable future tokens for unavailable covariates. These tokens interact with endogenous tokens and observed patches within the shared encoder. An auxiliary covariate prediction objective then supervises their future representations using observations from completed training windows. For unavailable covariates, these observations serve exclusively as training labels. 
%Target and covariate futures are decoded from their respective contextualized representations, with information exchange occurring before prediction. At inference, only historical observations and available future covariates are required.
