# reward_oracle.py
# 在 _predict_structure_esmfold 和 _calculate_affinity_pyrosetta 这两个核心函数中，我使用了伪代码和模拟输出来展示逻辑！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
# 需要做的如下：
    # 将目标PDB加载成一个PyRosetta的 'pose' (姿态) 对象。
    # 加载湿实验数据，并将其转换为与奖励分数兼容的格式。
    # 调用ESMFold来预测结构，并返回PDB文件路径和平均pLDDT分数。
    # 整个亲和力测试部分，_calculate_affinity_pyrosetta(self, binder_pdb_path: str):，需要pyrosetta计算的ΔΔG，以及根据这玩意算出来的loss function
    # 没了

import torch
import os
import pandas as pd
import numpy as np

# [API/库导入] - 你需要在这里导入你将使用的特定库
import esm
import pyrosetta
from pyrosetta import rosetta

from config import * # 从我们的配置文件中导入所有路径和权重

class RewardOracle:
    def __init__(self):
        """
        初始化奖励预言机。
        这包括一次性的设置任务，如加载目标结构和实验数据。
        """
        print("正在初始化奖励预言机(RewardOracle)...")
        
        # --- 1. 初始化 PyRosetta ---
        # [实现细节] 这个命令初始化PyRosetta。其标志可以自定义。
        # '-j' 指定用于并行任务的CPU数量。
        # '-ignore_unrecognized_res' 有助于避免处理非标准残基时程序崩溃。
        try:
            pyrosetta.init(f"-j {PYROSETTA_MAX_CPUS} -ignore_unrecognized_res")
            print("PyRosetta 初始化成功 (模拟状态)。")
        except Exception as e:
            print(f"严重错误: PyRosetta 初始化失败: {e}")

        # --- 2. 加载 GZMK 目标结构 ---
        # [实现细节] 将目标PDB加载成一个PyRosetta的 'pose' (姿态) 对象。
        # 这个pose对象将用于每一次的结合计算。
        if not os.path.exists(GZMK_TARGET_PDB_PATH):
            raise FileNotFoundError(f"在以下路径未找到目标PDB文件: {GZMK_TARGET_PDB_PATH}")
        # self.target_pose = pyrosetta.pose_from_pdb(GZMK_TARGET_PDB_PATH)
        self.target_pose_placeholder = "GZMK_POSE_LOADED" # 使用占位符表示已加载
        print(f"已从 {GZMK_TARGET_PDB_PATH} 加载目标GZMK结构。")

        # --- 3. 加载并处理湿实验数据 (@Terry) ---
        self.wet_lab_data = self._load_and_process_wet_lab_data(WET_LAB_DATA_PATH)
        print(f"已从湿实验数据中加载 {len(self.wet_lab_data)} 条记录。")

        # ... (PyRosetta 和湿实验数据的代码保持不变) ...
        
        # --- 新增: 加载ESMFold模型 (一次性加载) ---
        print("正在加载ESMFold模型到GPU...")
        try:
            # 加载ESMFold v1预训练模型
            self.esmfold_model = esm.pretrained.esmfold_v1()
            # 设置为评估模式，这会禁用Dropout等训练特有的层
            self.esmfold_model = self.esmfold_model.eval().cuda()
            print("ESMFold模型加载成功。")
        except Exception as e:
            raise RuntimeError(f"致命错误: 无法加载ESMFold模型，请检查fair-esm安装和CUDA环境: {e}")

        # ... (其余的 __init__ 代码) ...


    
    def _parse_plddt_from_pdb(self, pdb_string: str) -> float:
        """
        一个辅助函数，用于从PDB文件内容的字符串中解析出pLDDT分数。
        pLDDT分数存储在B-factor列中。
        """
        plddt_scores = []
        for line in pdb_string.split('\n'):
            if line.startswith('ATOM'):
                try:
                    # B-factor列是第11个字段，索引为10，但通常按固定宽度列解析更稳妥
                    # 字符位置 61-66 是B-factor
                    b_factor_str = line[60:66].strip()
                    plddt_scores.append(float(b_factor_str))
                except (ValueError, IndexError):
                    # 忽略无法解析的行
                    continue
        
        if not plddt_scores:
            return 0.0
        return np.mean(plddt_scores)

    def _load_and_process_wet_lab_data(self, path):
        """
        [统一湿实验数据与干实验性质预测 @Terry]
        加载实验数据，并将其转换为与奖励分数兼容的格式。
        """
        if not os.path.exists(path):
            print(f"警告: 在 {path} 未找到湿实验数据。将在没有湿实验数据的情况下继续。")
            return pd.DataFrame() # 如果文件不存在，返回空的DataFrame

        df = pd.read_csv(path) # 假设CSV文件包含 'sequence' 和 'Kd_nM' 两列
        
        # --- 数据统一化逻辑 ---
        # 目标: 将一个“越低越好”的指标 (如Kd) 转换为一个“越高越好”的奖励分数。
        # 一个常用的方法是对数尺度转换。
        # 我们添加一个很小的epsilon以避免计算log(0)。
        df['wet_lab_reward_score'] = -np.log10(df['Kd_nM'] + 1e-9)
        
        # 将序列设置成索引，以实现超快速查找 (平均时间复杂度O(1))
        return df.set_index('sequence')

    def _predict_structure_esmfold(self, sequence: str, sequence_id: str):
        """
        [已填充: 结构预测]
        调用ESMFold来预测结构，并返回PDB文件路径和平均pLDDT分数。
        """
        output_pdb_path = os.path.join(OUTPUT_DIR, f"{sequence_id}_pred.pdb")
        
        print(f"信息: 正在为序列 {sequence_id} 运行ESMFold预测...")
        try:
            # 使用在__init__中加载的模型进行预测
            # 我们不需要计算梯度，所以使用torch.no_grad()来节省内存
            with torch.no_grad():
                pdb_output_string = self.esmfold_model.infer_pdb(sequence)

            # 将预测结果（一个字符串）写入PDB文件
            with open(output_pdb_path, "w") as f:
                f.write(pdb_output_string)
            
            # 从PDB字符串中解析pLDDT分数
            # 除以100，因为ESMFold的pLDDT分数范围是0-100，而我们通常在0-1的尺度上讨论
            mean_plddt = self._parse_plddt_from_pdb(pdb_output_string) / 100.0

            print(f"信息: 序列 {sequence_id} 预测完成。PDB保存在 {output_pdb_path}。平均pLDDT: {mean_plddt:.3f}")
            
            return output_pdb_path, mean_plddt
            
        except Exception as e:
            print(f"错误: 对序列 {sequence_id} 的ESMFold预测失败: {e}")
            return None, 0.0 # 返回一个清晰的失败信号

    
    def _calculate_affinity_pyrosetta(self, binder_pdb_path: str):
        """
        [部署: 亲和力评估] & [Loss Function设计: 引入∆∆G]
        使用PyRosetta计算一个结合亲和力分数 (作为∆∆G的代理)。
        """
        # [实现细节] 这是一个高度简化的PyRosetta工作流程。
        # 一个真实的工作流会包含结构松弛、对接和一个完整的计算协议。
        # ------------ START OF 伪代码 ------------
        try:
            # 1. 加载由ESMFold预测的结合剂结构
            # binder_pose = pyrosetta.pose_from_pdb(binder_pdb_path)
            
            # 2. (复杂步骤) 将结合剂对接到目标pose上。
            # 这可以通过PatchDock服务器API后接RosettaDock等协议完成。
            # 为简单起见，我们假设它们已经形成了一个复合物。
            
            # 3. 使用InterfaceAnalyzer来获取结合能(dG)
            # 这是一个结合亲和力的代理指标。dG越负越好。
            # interface_analyzer = rosetta.protocols.analysis.InterfaceAnalyzer(complex_pose)
            # interface_analyzer.apply(complex_pose)
            # dG = interface_analyzer.get_interface_dG()
            
            # 在这个模板中，我们模拟输出
            print(f"模拟: PyRosetta亲和力计算...")
            dG = -np.random.uniform(5, 25) # 模拟一个负的dG值

            # 我们需要的是奖励分数，所以越高越好。返回dG的负值。
            return -dG 
        except Exception as e:
            print(f"错误: 对 {binder_pdb_path} 的PyRosetta计算失败: {e}")
            return -100.0 # 返回一个巨大的惩罚
        # ------------- END OF 伪代码 -------------

    def compute_reward(self, sequence: str) -> float:
        """
        主要的公共方法。为单个序列计算最终的奖励分数。
        """
        sequence_id = sequence[:10] # 使用前10个残基作为日志记录的唯一ID
        
        # --- 1. 首先检查湿实验数据库 (@Terry) ---
        if not self.wet_lab_data.empty and sequence in self.wet_lab_data.index:
            wet_score = self.wet_lab_data.loc[sequence]['wet_lab_reward_score']
            print(f"信息: 在湿实验数据中找到序列 {sequence_id}。分数为: {wet_score:.2f} + 额外奖励")
            # 对匹配已知有效结合剂的序列给予显著的额外奖励
            return wet_score + REWARD_WET_LAB_BONUS

        # --- 2. 如果不在数据库中，运行干实验预测流程 ---
        try:
            # 2a. 结构预测
            binder_pdb_path, mean_plddt = self._predict_structure_esmfold(sequence, sequence_id)
            if binder_pdb_path is None:
                return -10.0 # 惩罚结构预测失败的序列

            # [Loss Function设计: 引入蛋白质性质指标 (pLDDT)]
            # 将pLDDT既作为直接的奖励组件，也作为置信度过滤器。
            plddt_reward = mean_plddt

            # 2b. 亲和力计算
            affinity_reward = self._calculate_affinity_pyrosetta(binder_pdb_path)

            # --- 置信度加权 ---
            # 如果pLDDT分数很低，说明结构不可靠，那么亲和力计算也可能没有意义。
            # 我们应该降低它的权重。
            if mean_plddt < 0.7:
                print(f"警告: 序列 {sequence_id} 的pLDDT置信度低 ({mean_plddt:.2f})。将惩罚其亲和力分数。")
                affinity_reward *= 0.1 # 如果结构很差，亲和力分数将大打折扣

            # --- 3. 最终的组合奖励 ---
            # 所有组件的加权总和。权重从config.py中导入。
            final_reward = (
                REWARD_W_DDG * affinity_reward +
                REWARD_W_PLDDT * plddt_reward
            )
            
            print(f"信息: 序列 {sequence_id} 的干实验奖励: 亲和力={affinity_reward:.2f}, pLDDT={plddt_reward:.2f} -> 最终奖励={final_reward:.2f}")
            return final_reward

        except Exception as e:
            print(f"致命错误: 在为序列 {sequence_id} 计算奖励时发生未处理的错误: {e}")
            return -20.0 # 对任何意外崩溃给予巨大的惩罚
