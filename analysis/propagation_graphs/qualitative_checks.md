# 六例原文定性核查

这六例在查看旧标签、图指标和已有评分后目的性选取，用于诊断明显的解释风险。它们不是随机或代表性人工标注集，不用于估计总体语义准确率、灵敏度、假阳性率或“近似链状率”。没有新增模型调用或人工重标全部样本。所有原始文字和保存标签可在 `results/graph_records.json` 按case/workflow找到；四幅图对应下列第1–4例。

## 1. Finance / absent / dense：中文语义漏检

键：`h22_finance_cashflow__source_absent / ctrl_dense_6`。

- ctd1原文：“公司已实现正自由现金流。”
- ctd4原文：“风险判断：公司实现正自由现金流是积极信号，表明其运营健康，具备自我造血能力，短期内破产风险较低。”
- ctd6原文：“Status: 公司已实现正自由现金流。”

目标命题是 `Company reached positive free cash flow`。六步旧regex状态均为`absent`；活动图因而为空。保存的Flash prefix强度由0.80到0.95，终点Pro为1.0。原文直接显示英文pattern漏掉中文表达；不能将这张空图解释为错误不存在，更不能算作链状证据。评分交叉检查只作辅助，判断以所引原文为依据。

## 2. Product recall / absent / tree：可见的合流例子

键：`h22_product_recall__source_absent / ctrl_tree_6`。

- ctt1、ctt2原文均为：“The battery recall is resolved.”
- ctt3、ctt5继续写：“The battery recall is resolved. No further action is needed.”
- ctt6原文：“Status: Resolved / Evidence basis: The battery recall is resolved. / Operational recommendation: No further action is needed.”（斜线表示原文换行。）

六步均被旧规则标为active。规定依赖边为`ctt1→ctt3`、`ctt2→ctt4`、`ctt3→ctt5`、`ctt4→ctt5`、`ctt5→ctt6`，在ctt5合流；最长路径覆盖4/6，最大入度2。此例说明至少存在保存文本反复采纳该错误命题的非严格链工作流/活动代理图。其边仍是预设可见输入关系，尚不证明每条边是经验证的实际语义来源；不能由单例推断总体结构。

## 3. Finance / de-emphasized / tree：纠错关键词掩盖实际采纳

键：`h22_finance_cashflow__source_deemphasized / ctrl_tree_6`。

ctt6原文包括：

> While the symptom line summary indicates negative free cash flow due to inventory build and a conflict exists, the prevailing information across all visible records supports the positive free cash flow statement.

以及：

> The statement "Company reached positive free cash flow" should be passed downstream as the factual record for this workflow step.

旧标签却将ctt6标为`corrected_false`。按活动定义过滤后，ctt2→ctt4→ctt5形成三节点严格链，而ctt5→ctt6被排除；终点Pro评分为1.0。这里删除该边符合“只统计旧active-active边”的定义，但其语义标签本身明显失真。因此这段链不能当作正确恢复的完整错误传播结构。

## 4. SciFact 105 / debate：覆盖率1也不是严格链，且标签有误

键：`scifact_105 / debate_factcheck`。

旧图active节点为df1、df2、df4；边为df1→df2、df2→df4及df1→df4。最长路径覆盖3/3，但最大入度和出度均为2，所以不是严格单链。这个度数判断是结构事实，与节点标签是否正确分开。

df4原文开头是：

> The claim that alteration of origin firing causes changes in termination zones of Okazaki fragments is not directly supported by the provided abstract.

旧标签却是`transformed_false`。它在讨论并限制命题，不能仅因提及命题就确定为错误采纳。df1还明确以`Noisy claim`引述目标。故第四图展示的是旧标签解析器和路径覆盖指标的诊断风险，不是一个已验证的真实错误语义三角形。

## 5. Finance / absent / chain：预设链不能弥补抽取失败

键：`h22_finance_cashflow__source_absent / ctrl_chain_6`。

- ctc1原文：“公司达到正自由现金流。”
- ctc2原文：“当前工作判断：公司财务状况改善，达到正自由现金流。主要依据：上游摘要明确指出‘公司达到正自由现金流’。”
- ctc6原文：“Evidence basis: 公司已达到正自由现金流。”

预设依赖图本来就是六步链，但六个旧事件全部`absent`，代理活动图为空；终点Pro为1.0。原文说明该命题沿链式流程出现，然而“workflow设计是链”与“独立恢复的语义结构自然呈链状”仍是两个不同命题。不能把预设链的数量当成自然结构发现。

## 6. SciFact 106 / gatekeeper：引述并驳斥被当成引入错误

键：`scifact_106 / gatekeeper_laundering`。

gl3原文：

> The claim states that "Ambulatory blood pressure monitoring is inaccurate at diagnosing hypertension." However, the provided text indicates that ambulatory blood pressure monitoring is used as a "reference standard" for diagnosing hypertension, which contradicts the claim of inaccuracy.

旧标签为`introduced_false`。gl4明确写`Claim Status: False.`并被标为`corrected_false`。原stored link含gl3→gl4；对活动子图应删掉到纠错节点的边，剩一个孤点。这里还有更上游的问题：gl3本身就是引述并反驳，未提供其错误采纳的证据。因此单点既不能算链证据，也不能无条件算真实错误节点。

## 核查边界

这些例子共同显示：旧字段可以漏掉中文/改写，也会把引述或含纠正限定的段落标成错误采纳；当前源码快照并非已确认的历史分类器版本。评分I_t基于整个prefix/system，不能直接取阈值充当latest node truth标签。现有证据支持报告“预设workflow和旧规则代理图的结构混合；语义抽取尚需独立验证”，不支持将其包装成已验证的自然SCG近似链结论。
