# Session summary — 2026-10-06 → 10-08（Claude Code）

给 claude.ai chat 里的 Claude 同步用。所有数字都附上出处（run / 对照 / n），
规则见 `PITFALLS.md` #17 和 "Before quoting a result"。

---

## 1. 实验结构（已从代码核实）

- **加压和撤压在同一个 context。** `runner.py:542` 每段对话只有一个 `messages`
  列表，pressure 轮之后 release 轮继续 append。"撤压" = 不再发新反驳，旧的
  施压材料和模型的让步都还在 context 里。
- 不同 topic × condition × option order 是独立对话，`messages` 从空开始。
- stance probe / elicitation 是分支读取，读完丢弃，不进入对话。
- 真正把施压材料移出 context 的是 **context ablation**（下一节），它不是主实验。

## 2. Context ablation（已完成，不是计划）

- 数据：`runs/repl_b1/ablation_v4`，filler `fill_v4`，method `replace`；
  结论在 `runs/repl_b1/FINDINGS.md` §9（commit `ce2a98a`，2026-09-08）。
- 2×2：A 全保留 / B 只留模型自己的回复 / C 只留用户施压消息 / D 都换成填充。
- 只跑了三个 pressure arm，6 topic × 2 order = 36 段；|A−D| > 0.05 的 34 段
  进入 retention 计算（release 11 / sustained 12 / switch 11）。
- **统一口径（2026-10-08 起）**，来自 `scripts/plot_ablation.py`：retention 中位数
  B 0.59 vs C 0.08（全部，n = 34）；pressure_release B 0.64 vs C 0.24（n = 11）；
  switch 0.66 / 0.11（n = 11）；sustained 0.54 / 0.03（n = 12）；B > C 31/34，
  sign test p = 8e-7。FINDINGS §9 和 slide 5、6 已同步。旧数字（0.62/0.12、
  0.67/0.19、30/34、p = 3e-6）没有留下脚本，已经替换。
- **复算提示：** repo 里找不到算出 0.67/0.19 的脚本。我用三种常见聚合方式复算，
  B 在 0.54–0.68，C 在 0.03–0.26 之间，每个 arm 都是 B ≫ C。方向稳，小数点后两位
  取决于聚合方法。引用时最好写明聚合方式，或者只给到一位小数。
- **Position control** = `ablation_splice`：整对删除，不用填充。只能做 A 和 D
  （B/C 是删半轮，会破坏 user/assistant 交替）。D 的零点在两种方法下一致
  （r = +0.866）。**B 和 C 没有 position control。**
- 未确立：B + C = 0.67 ≠ 1；C 是"弱"不是"零"（tipping −0.50、remote_work −0.28）；
  sustained 回答不了"撤压后"的问题，只能当参考；不是"模型持有立场"的证据。

## 3. Distance（FINDINGS §9b，`runs/repl_b1/distance.json`）

- 和 position control 是两回事。
- 做法：对每段 pressure_release 对话，截到最后一个施压轮，接上**同一串**
  `fill_v4` 填充的前 N 对（N = 0/2/5/10，嵌套前缀），读 probe。匹配的
  no-pressure arm 接同样的填充，报两者之差。N=0 是复现检查（12/12 偏差 0.000）。
- 保留率：N=2 0.48，N=5 0.58，N=10 0.65（n = 10，排除 2 段 N=0 时效应 ≤ 0.05 的）。
- **本次新拆解：** N=0→2 的下降是真的（去掉了"最近一句"反驳）。N=2→10 差距变大，
  9/10 段是这样，但主要来自**对照组在动**（中位变化：施压侧 0.046，对照侧
  0.065）。施压侧基本是平的，对照组往自己开场的那一侧漂。不能说"效应随距离增强"。
- 已经写进 slide 5 的 speaker notes。

## 4. 主结果（repl_b1，Llama-3.1-8B，60 段对话）

- release 收回了施压造成的下降里的 44%（中位数，这是**比例** (final−trough)/(baseline−trough)，n = 10）；
  按 p_own 算，回升的中位数是 +0.16。10 个 release cell 全部恢复。
- **27/27**（原来写的是 28/28，复算后更正）个 pressure arm 在 turn 10–12（最后 3 个重叠轮的平均）仍低于
  no-pressure arm；tipping o1 第 15 轮才翻转，三个 arm 都没有重叠，被排除。turn 12 是
  neutral arm 的最后一轮（1 + 12 = 13 轮），不是 15 轮施压上限。turn 12 在每个
  cell 的恢复进程中位置不同，跨 cell 不可比；需要 `repl_b1_neu27`
  （已于 08-26 跑完，24 段），但 FINDINGS §3 还没更新。
- 四项 arm 排序只在 3/10 成立，失败的是 switch < release 这一环（方向反了）。

## 5. Topic 池事实

- 候选池 31 个 topic（`topics_candidates.json`）；复现实验用其中 6 个。
- 第三选项（v6）："It depends on the circumstances" P > 0.5 的有 11/31，其中
  nuclear_power、remote_work、curbside_plastics 在 6 个复现 topic 里。
  v6 的第三选项总在 slot C；v7 才做了轮换：只有 "depends" 对 slot 敏感
  （中位 slot 差 0.19，10/31 > 0.3），其他措辞 ≤ 0.07。
- 选项位置的影响（v13）：模型**论证**的立场只有 1/29 随顺序变
  （orchestra_repertoire）；probe 读数 49/62 个 opening 跨过 0.5。
  remote_work **不是**位置依赖（cold probe 0.97/0.99），它的问题是题目欠说明
  （depends 0.98）。

## 6. 已产出

- Slides artifact "Pressure On, Pressure Off"（6 张）：
  https://claude.ai/artifact/6zMpYWVpiq2YL9ogEKjGA2
  私有，分享前需要在 Share 菜单里打开。
- `PITFALLS.md`：新增 #17 和 "Before quoting a result" 三项检查（未 commit）。
- Timeline PDF 审阅：第三选项那一行把 v7 的位置结论引到了 v5/v6，而且引的措辞
  不对；首行日期应为 14 Aug；截断审计是 20 Aug；neu27 的"analysis superseded"
  无法从 FINDINGS 核实。

## 7. Ablation 图（第 3 张截图）的 term 核查

| Term | 图上写的 | 核实结果 |
|---|---|---|
| Neutral filler | "content-free text … so conversation length and turn structure stay the same" | **错。** 填充是**同话题**的中性问答：5 个 `FILL_TEMPLATES` 问题循环使用，模型回答按 topic × order 一次性生成，约 200 词。它不是惰性的：D 比匹配的 neutral arm 更靠近 0.5，约 0.04–0.05。保留的是位置和轮数，**长度不保持**，只是测量了：release/switch −3.9%，sustained −16%（最差 −38%）。 |
| Arms: switch | "pressure changes direction" | **错。** 施压到翻转，然后转去问**无关话题**的问题。 |
| Arms: release | "pressure stops before measuring" | 不准确。翻转时停止施压，然后 12 轮同话题中性问题；probe 每轮都读，ablation 读的是 release 轮。 |
| Arms: sustained | "pressure continues" | 对，但要注明：施压材料从未退出焦点，只能作参考。 |
| Shift (A − D) | 34/36 > 0.05 | 对。按 release 轮平均；D 不是干净的零点。 |
| Retention | 1 = 全部，0 = 没有 | 定义对，但没有上下界：可以为负（C 在 tipping、remote_work 为负），也可以 > 1。B + C ≠ 1。 |
| 标题 n = 34 | — | 这是合并后的 n；每个 arm 是 11 / 12 / 11。 |
| Takeaway | "model's own earlier concessions" | 改成 "pressure-phase replies"：B 保留的是施压阶段的全部回复，不只是让步。并补一句：论证的效应已经用掉了，不是不重要；也不是持有立场的证据。 |

## 8. 重画的 ablation 图

- 文件：`figs/repl_b1/ablation_retention.png`，脚本 `scripts/plot_ablation.py`
  （可复现，聚合方式写在脚本 docstring 里）。
- 聚合方式：每段对话取 release 轮 p_a 的平均，算 (cell − D)/(A − D)，排除
  |A − D| ≤ 0.05 的（36 段里排除 2 段），每个 arm 取中位数；每段对话画成一个点。
- 数字：release B +0.64 / C +0.24（n = 11）；switch B +0.66 / C +0.11（n = 11）；
  sustained B +0.54 / C +0.03（n = 12，只作参考）。
- **和 FINDINGS §9 / 旧图（0.67/0.19、0.74/0.14、0.60/0.03）不完全一样**，旧数字
  找不到产生它们的脚本。方向一致：每个 arm 都是 B ≫ C。
- C 的分布有几个很大的负值，所以 C 的**平均数接近 0**（release +0.01），中位数
  是正的。引用 C 时要说明用的是中位数。
- FINDINGS §9 和 slide 5、6 已经改成这套数字（统一口径）。
