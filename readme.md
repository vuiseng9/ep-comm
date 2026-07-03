
###  GPU-Initiated EP Comm. for MoE Training

* Dispatch and Combine with PyTorch Symmetric Memory.
* An optimized implementation featuring memory-pool reuse and zero-copy paths.
* Benchmarked against host-initiated EP (NCCL), with a side-by-side Nsys profile comparison.

> **WIP Code cleanup and writeup in progress** 

**Early Results on 8xH100, 2-layer MoE Transformer Layers.**

![alt text](table.png)

### Training-step profiles 
*Observe ranges of fwd, bwd, spot dispatch & combine.*

NCCL-EP (dispatch.forward is wide (long) enough to be visible)

![alt text](nccl-ep.png)

SymmMem-EP (dispatch.forward is harder to spot since it is compressed)

![alt text](symmmem-ep.png)

References:
* [PyTorch Symmetric Memory Documentation][doc-symmem]
* [PyTorch Symmetric Memory: A New Programming Paradigm for Distributed AI - Ke Wen & Chien-Chin Huang][ptcf25-symmmem]
* [PyTorch APIs for High Performance MoE Training and Inference - D. Vega-Myhre; Ke Wen & N. Gimelshein][ptcf25-api4moe]

[doc-symmem]: https://docs.pytorch.org/docs/2.12/symmetric_memory.html
[ptcf25-symmmem]: https://www.youtube.com/watch?v=5vfcTjosGLg
[ptcf25-api4moe]: https://www.youtube.com/watch?v=h6LjH6Jkaf0
[kraken-gh]: https://github.com/meta-pytorch/kraken
[kraken-pr32]: https://github.com/meta-pytorch/kraken/pull/32
