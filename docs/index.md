## Abstract

This paper investigates how to efficiently transition and update policies, trained initially with demonstrations,  using off-policy actor-critic reinforcement learning. It is well-known that techniques based on Learning from Demonstrations, for example behavior cloning, can lead to proficient policies given limited data. However, it is currently unclear how to efficiently update that policy using reinforcement learning as these approaches are inherently optimizing different objective functions. Previous works have used loss functions which combine behavioral cloning losses with reinforcement learning losses to enable this update, however, the components of these loss functions are often set anecdotally, and their individual contributions are not well understood. In this work we propose the Cycle-of-Learning (CoL) framework that uses an actor-critic architecture with a loss function that combines behavior cloning and 1-step Q-learning losses with an off-policy pre-training step from human demonstrations. This enables transition from behavior cloning to reinforcement learning without performance degradation and improves reinforcement learning in terms of overall performance and training time. Additionally, we carefully study the composition of these combined losses and their impact on overall policy learning. We show that our approach outperforms state-of-the-art techniques for combining behavior cloning and reinforcement learning for both dense and sparse reward scenarios. Our results also suggest that directly including the behavior cloning loss on demonstration data helps to ensure stable learning and ground future policy updates.

## Video

To be uploaded.

## Citation

You can find our complete paper on arXiv (https://arxiv.org/abs/1810.11545). Please cite our work as  
```
@inproceedings{goecks2019efficiently,
  title={Efficiently combining human demonstrations and interventions for safe training of autonomous systems in real time},
  author={Goecks, Vinicius G and Gremillion, Gregory M and Lawhern, Vernon J and Valasek, John and Waytowich, Nicholas R},
  booktitle={AAAI Conference on Artificial Intelligence (2019). Frames/sec vs Params No GPU},
  volume={140},
  year={2019}
}
```

## Acknowledgments

Research was sponsored by the U.S. Army Research Laboratory and was accomplished under Cooperative Agreement Number W911NF-18-2-0134. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Army Research Laboratory or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes not withstanding any copyright notation herein.
