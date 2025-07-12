# config.py
import torch

# ---- 模型与路径配置 ----
EVODIFF_MODEL_NAME = "esm_if1_gvp4_t16_142M_UR50" # 示例，使用你的EvoDiff模型
GZMK_TARGET_PDB_PATH = "data/GZMK_target.pdb"
WET_LAB_DATA_PATH = "data/wet_lab_data.csv"
OUTPUT_DIR = "outputs/"

# ---- GRPO 训练超参数 ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-5
NUM_EPOCHS = 100
GROUP_SIZE_K = 16  # 每个prompt生成16个候选序列
KL_BETA = 0.1      # KL散度正则项的权重

# ---- 奖励函数权重 ----
# Reward = w1 * ddg_score + w2 * plddt_score + w3 * wet_lab_bonus
REWARD_W_DDG = 1.0       # 来自PyRosetta的∆∆G的权重
REWARD_W_PLDDT = 0.2     # 来自ESMFold的pLDDT的权重
REWARD_WET_LAB_BONUS = 5.0 # 对已验证有效的序列给予额外奖励

# ---- 计算资源配置 ----
# 根据你的硬件调整
PYROSETTA_MAX_CPUS = 8
