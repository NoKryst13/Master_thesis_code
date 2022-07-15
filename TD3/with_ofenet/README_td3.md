PyTorch implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3) with integrated OFENet. 

This implementation is based on the original implementation of TD3 by Fujimoto et al. [paper](https://arxiv.org/abs/1802.09477).

It includes the Online Feature Extractor Network (OFENet) by Ota et al. (https://arxiv.org/abs/2003.01629) 
to learn representations using different auxiliary tasks.


How to use and possible input parameters:

python main.py \\
	--env	\\
	--max_timesteps	\\
	--aux_task		\\
	--pretrain_steps	\\
	--total_units		\\
	--wandb_name		\\
	--seed			\\
	
Exemplary Run, when no wandb_name is specified wandb (Weights&Biases, https://www.wandb.com/) is not used at all:

python main.py \
	--env=Humanoid-v2 \
	--max_timesteps=3000000 \
	--aux_task=fsp \
	--pretrain_steps=10000 \
	--total_units=240 \
	--seed=0
	

