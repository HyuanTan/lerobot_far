# merge_phase3.py — 合并服务器 candidate 记录 + 客户端 episode 结果
# import pandas as pd

# cands  = pd.read_json("./mc_data/candidates.jsonl", lines=True)
# outcs  = pd.read_json("./mc_data/client_outcomes.jsonl", lines=True)
# merged = cands.merge(outcs, on="episode_id")

# # 每个 candidate 的平均成功率（按 server_selected_idx 分组）
# print(merged.groupby("server_selected_idx")["success"].mean())

# # 保存训练数据
# merged.to_parquet("./mc_data/phase3_training.parquet")


import pandas as pd

# 合并三张表
srv   = pd.read_json("mc_data/candidates.jsonl", lines=True)
steps = pd.read_json("mc_data/client_steps.jsonl", lines=True)
outcs = pd.read_json("mc_data/client_outcomes.jsonl", lines=True)

full = (
    srv.merge(steps, on=["episode_id", "timestep"])
       .merge(outcs, on="episode_id")
)

# P0：客户端覆盖率 vs 成功率
print(full.groupby("client_override")["success"].mean())

# P0：spread_l2 vs 成功率（模型不确定性分析）
full["spread_bin"] = pd.cut(full["candidate_spread_l2"], bins=5)
print(full.groupby("spread_bin")["success"].mean())

# P1：执行连续性 vs 成功率
print(full[["execution_continuity","success"]].corr())

# P1：不同 episode 阶段的 delay 选择分布
full["phase_bin"] = pd.cut(full["episode_phase"], bins=[0,.33,.66,1],
                           labels=["early","mid","late"])
print(full.groupby(["phase_bin","delay_selected"])["success"].mean())

# P2：robot_state 变化幅度分析
import numpy as np
full["state_norm"] = full["robot_state"].apply(
    lambda s: float(np.linalg.norm(s)) if s else None
)
print(full.groupby("success")["state_norm"].describe())
