# Master Thesis Code
Code I used for the experiments of my masters thesis on the topic "A Comparison of Auxiliary Tasks for Low-Dimensional Representation Learning".

I implemented the Online Feature Extractor Network (OFENet) by Ota et al. (https://arxiv.org/abs/2003.01629) in PyTorch and 
combined it with TD3 and SAC.

The Code for TD3 is based on the original Code by Fujimoto et al. (https://arxiv.org/abs/1802.09477)
The Code for SAC is based on the PyTorch implementation by Yarats et al. (https://github.com/denisyarats/pytorch_sac)

The folders TD3 and SAC each contain a version with and without OFENet including a separate readme, which explains how to use the particular version.

