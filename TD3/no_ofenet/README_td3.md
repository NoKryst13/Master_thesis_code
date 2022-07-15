PyTorch implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3) with integrated OFENet. 

This implementation is based on the original implementation of TD3 by Fujimoto et al. [paper](https://arxiv.org/abs/1802.09477).

How to use and possible input parameters:

python main.py \\
	--env	\\
	--max_timesteps	\\
	--wandb_name		\\
	--seed			\\
	
Exemplary Run, when no wandb_name is specified wandb (Weights&Biases, https://www.wandb.com/) is not used at all:

python main.py \
	--env=Humanoid-v2 \
	--max_timesteps=3000000 \
	--seed=0
	
