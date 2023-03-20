# Rethinking Efficacy of Softmax for Lightweight Non-Local Neural Networks

**This is the official repository of "Rethinking Efficacy of Softmax for Lightweight Non-Local Neural Networks", ICIP 2022.**

## Abstract

Non-local (NL) block is a popular module that demonstrates the capability to model global contexts. However, NL block generally has heavy computation and memory costs, so it is impractical to apply the block to high-resolution feature maps. In this paper, to investigate the efficacy of NL block, we empirically analyze if the magnitude and direction of input feature vectors properly affect the attention between vectors. The results show the inefficacy of softmax operation which is generally used to normalize the attention map of the NL block. Attention maps normalized with softmax operation highly rely upon magnitude of key vectors, and performance is degenerated if the magnitude information is removed. By replacing softmax operation with the scaling factor, we demonstrate improved performance on CIFAR-10, CIFAR-100, and Tiny-ImageNet. In Addition, our method shows robustness to embedding channel reduction and embedding weight initialization. Notably, our method makes multi-head attention employable without additional computational cost.

## Dependencies
Make sure you have the following dependencies installed before proceeding:
- Python 3+ distribution
- PyTorch == 1.4.0

## License
This project is distributed under [MIT license](LICENSE.md). If you use our code/models in your research, please cite our paper:
```
@inproceedings{cho2022rethinking,
  title={Rethinking Efficacy of Softmax for Lightweight Non-local Neural Networks},
  author={Cho, Yooshin and Kim, Youngsoo and Cho, Hanbyel and Ahn, Jaesung and Hong, Hyeong Gwon and Kim, Junmo},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  pages={1031--1035},
  year={2022},
  organization={IEEE}
}
```

