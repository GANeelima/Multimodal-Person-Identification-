# Multimodal-(Person-Identification)
Dataset can be downloaded from here: [link for dataset](https://sail.usc.edu/iemocap/)

Database used: IEMOCAP

Directory Structure:

![image](https://github.com/GANeelima/Multimodal-Person-Identification-/assets/114975668/d2cc140d-a5be-4d66-8bb4-ffecea6b2856)


### Environment:

* Python 3.7
* PyTorch 1.6.0
* torchaudio 0.6.0
* torchvision 0.7.0
* transformers 3.4.0 (huggingface)
* sparseconvnet

  for sparseconvnet, clone the repository:
  ```console
  !git clone https://github.com/facebookresearch/SparseConvNet
  ```
  and
  run this command:
   ```console
  !bash develop.sh
  ```
  
* facenet-pytorch 2.3.0
* sentencepiece
  
  ```console
  !pip install sentencepiece
  ```
#### Creating venv in colab:

```console
!pip install -q condacolab
import condacolab
condacolab.install()
```
### Train the MME2E

```console
python main.py -lr=5e-5 -ep=40 -mod=tav -bs=8 --img-interval=500 --early-stop=6 --loss=bce --cuda=3 --model=mme2e --num-persons=10 --trans-dim=64 --trans-nlayers=4 --trans-nheads=4 --text-lr-factor=10 --text-model-size=base --text-max-len=100
```

### Train the sparse MME2E

```console
python main.py -lr=5e-5 -ep=40 -mod=tav -bs=2 --img-interval=500 --early-stop=6 --loss=bce --cuda=3 --model=mme2e_sparse --num-persons=10 --trans-dim=64 --trans-nlayers=4 --trans-nheads=4 --text-lr-factor=10 -st=0.8 --text-model-size=base --text-max-len=100
```

For reference, the github link for Multimodal-End to End Sparse Model is given here:
[link](https://github.com/wenliangdai/Multimodal-End2end-Sparse.git/)
