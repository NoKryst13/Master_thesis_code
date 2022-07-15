PyTorch implementation of Soft Actor Critic (SAC) (https://arxiv.org/abs/1801.01290) with integrated OFENet. 

This implementation is based on the implementation by Yarats et al. (https://github.com/denisyarats/pytorch_sac)

It includes the Online Feature Extractor Network (OFENet) by Ota et al. (https://arxiv.org/abs/2003.01629) 
to learn representations using different auxiliary tasks.


How to use and possible input parameters:

python train.py \\
	--env	\\
	--num_train_steps	\\
	--wandb_name		\\
	--seed			\\
	
Exemplary Run, when no wandb_name is specified wandb (Weights&Biases, https://www.wandb.com/) is not used at all:

python train.py \
	env=Humanoid-v2 \
	num_train_steps=3000000 \
	seed=0
	

