# evodiff_GRPO_GZMK


待完成：
  1. 对接PyRosetta的API，计算亲和力
  2. 获得GZMK处理后的分子结构
  3. 湿实验数据表格

文件架构
/gzmk_binder_design/
|-- config.py                 # 存放所有超参数和路径
|-- models.py                 # EvoDiff模型封装
|-- reward_oracle.py          # 核心：奖励计算模块 (ESMFold, PyRosetta, 湿实验)
|-- grpo_trainer.py           # GRPO训练器主框架
|-- main.py                   # 主程序入口
|-- data/
|   |-- GZMK_target.pdb       # 目标蛋白GZMK的PDB文件
|   |-- wet_lab_data.csv      # 湿实验数据表格
