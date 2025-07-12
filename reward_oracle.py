# reward_oracle.py
import pandas as pd
import pyrosetta
from config import *

class RewardOracle:
    def __init__(self):
        # 初始化PyRosetta
        # [实现细节] 根据你的系统配置PyRosetta初始化参数
        pyrosetta.init(f"-j {PYROSETTA_MAX_CPUS}")

        # 加载目标结构
        self.target_pose = pyrosetta.pose_from_pdb(GZMK_TARGET_PDB_PATH)

        # 加载并处理湿实验数据 (@Terry)
        self.wet_lab_df = self.load_and_process_wet_lab_data(WET_LAB_DATA_PATH)

    def load_and_process_wet_lab_data(self, path):
        """
        统一湿实验数据与干实验性质预测 (@Terry)
        - 读取湿实验数据
        - 将实验结果 (如Kd, IC50) 映射到一个与计算指标可比的数值尺度上
        """
        df = pd.read_csv(path) # 假设CSV有 'sequence' 和 'Kd_nM' 两列
        # [实现细节] 设计一个映射函数
        # 例如，将Kd值转换为一个奖励分数，Kd越小分数越高
        # log-scale mapping: score = -log10(Kd_nM)
        df['wet_lab_score'] = -np.log10(df['Kd_nM'])
        return df.set_index('sequence')

    def predict_structure(self, sequence):
        """使用ESMFold或OmegaFold预测结构，并返回pLDDT"""
        # [实现细节] 调用你的结构预测模型API
        # 这里是伪代码
        # pdb_string, plddt_scores = esmfold_api.predict(sequence)
        # mean_plddt = np.mean(plddt_scores)
        # temp_pdb_file = f"{OUTPUT_DIR}/{sequence[:10]}.pdb"
        # with open(temp_pdb_file, "w") as f:
        #     f.write(pdb_string)
        # return temp_pdb_file, mean_plddt

        # 模拟返回
        print(f"INFO: Simulating structure prediction for {sequence[:10]}...")
        return "path/to/simulated.pdb", 0.85

    def calculate_ddg_pyrosetta(self, binder_pdb_path):
        """引入∆∆G等来自于PyRosetta等亲和力指标"""
        # [实现细节] 实现PyRosetta的∆∆G计算流程
        # 这是一个高度简化的流程
        # 1. 加载binder和target，组合成复合物
        # binder_pose = pyrosetta.pose_from_pdb(binder_pdb_path)
        # complex_pose = self.form_complex(self.target_pose, binder_pose)
        # 2. 使用Rosetta的ddG协议
        # ddg_filter = pyrosetta.rosetta.protocols.analysis.InterfaceAnalyzer(complex_pose)
        # ddg_value = ddg_filter.get_interface_dG() # dG是结合自由能，其变化量∆∆G更有意义
        # 为了简化，我们直接使用dG的负值作为亲和力分数
        # return -ddg_value

        # 模拟返回
        print(f"INFO: Simulating PyRosetta ddG calculation...")
        return 15.0 # 返回一个模拟的亲和力分数

    def compute_reward(self, sequence):
        """计算单个序列的最终奖励"""
        # 1. 检查湿实验数据库 (@Terry)
        if sequence in self.wet_lab_df.index:
            print(f"INFO: Found sequence {sequence[:10]} in wet lab data!")
            wet_score = self.wet_lab_df.loc[sequence]['wet_lab_score']
            # 对已验证的序列给予高额固定奖励，或使用其映射分数
            return wet_score + REWARD_WET_LAB_BONUS

        # 2. 如果不在湿实验库中，进行干实验预测
        try:
            # 2a. 结构预测
            binder_pdb_path, mean_plddt = self.predict_structure(sequence)

            # 2b. 亲和力评估
            ddg_score = self.calculate_ddg_pyrosetta(binder_pdb_path)

            # 2c. 引入蛋白质性质指标 (如pLDDT)
            # 对低质量结构进行惩罚
            plddt_score = mean_plddt
            if mean_plddt < 0.7:
                ddg_score *= 0.5 # 结构不可信，亲和力打折扣

            # 3. 组合成最终奖励
            reward = (REWARD_W_DDG * ddg_score) + (REWARD_W_PLDDT * plddt_score)
            return reward

        except Exception as e:
            print(f"ERROR: Reward calculation failed for sequence {sequence[:10]}: {e}")
            return -10.0 # 对计算失败的序列给予极低的奖励

