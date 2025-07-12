# main.py
from grpo_trainer import GRPOTrainer

if __name__ == "__main__":
    # 定义设计任务的提示
    # 对于蛋白质设计，prompt可以很简单，比如只指定长度
    training_prompts = [
        {"length": 100},
        {"length": 110},
        {"length": 120},
        # 你可以定义更复杂的prompt，例如包含一个已知的结合motif作为scaffold
    ]

    # 初始化并开始训练
    trainer = GRPOTrainer()
    trainer.train(training_prompts)

    print("GRPO training finished.")
