# grpo_trainer.py
import torch
from torch.optim import Adam
from config import *
from models import EvoDiffPolicyModel
from reward_oracle import RewardOracle

class GRPOTrainer:
    def __init__(self):
        self.policy_model = EvoDiffPolicyModel(EVODIFF_MODEL_NAME, DEVICE)
        self.ref_model = EvoDiffPolicyModel(EVODIFF_MODEL_NAME, DEVICE) # 参考模型，不训练
        self.ref_model.model.eval()

        self.optimizer = Adam(self.policy_model.model.parameters(), lr=LEARNING_RATE)
        self.reward_oracle = RewardOracle()

    def train(self, prompts):
        for epoch in range(NUM_EPOCHS):
            for i, prompt in enumerate(prompts):
                # 步骤 1: 批量生成
                sequences = self.policy_model.generate_group(prompt, GROUP_SIZE_K)

                # 步骤 2: 计算奖励
                rewards = [self.reward_oracle.compute_reward(seq) for seq in sequences]
                rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)

                # 步骤 3: GRPO核心 - 计算优势
                group_baseline = torch.mean(rewards)
                advantages = rewards - group_baseline

                # 步骤 4: Loss Function设计
                log_probs_policy = self.policy_model.get_log_probs(sequences)

                with torch.no_grad():
                    log_probs_ref = self.ref_model.get_log_probs(sequences)

                # a. 策略损失 (Policy Gradient Loss)
                prob_ratio = torch.exp(log_probs_policy - log_probs_ref)
                policy_loss = -torch.mean(advantages * prob_ratio)

                # b. KL正则项 (KL Regularization)
                # 防止策略模型偏离原始模型太远，保证生成蛋白质的基本属性
                kl_div = torch.mean(log_probs_policy - log_probs_ref)

                # c. 总损失
                total_loss = policy_loss + KL_BETA * kl_div

                # 步骤 5: 优化
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Prompt {i+1} | Avg Reward: {group_baseline.item():.3f} | Loss: {total_loss.item():.3f} | KL: {kl_div.item():.3f}")

                # [可选] 保存模型
                if (i+1) % 10 == 0:
                     torch.save(self.policy_model.model.state_dict(), f"{OUTPUT_DIR}/policy_model_epoch_{epoch+1}.pt")
