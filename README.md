# Person Identification based on Multimodal analysis

In this project, we propose an innovative fully end-to-end model that effectively integrates feature extraction and optimization phases for person identification using multiple modalities of data from the IEMOCAP dataset. Our goal is to enhance the overall performance of multimodal affective computing tasks by jointly optimizing these steps.

To enable seamless end-to-end training, we carefully reorganize the IEMOCAP dataset, ensuring better alignment between different modalities and the target task of person identification. This reorganization facilitates more effective integration of information from various data sources, leading to improved accuracy in identifying individuals.

To address potential computational challenges associated with the end-to-end model, we introduce a novel sparse cross-modal attention technique for feature extraction. This technique allows us to focus on the most relevant and informative elements across modalities while reducing the computational complexity. As a result, the model becomes more efficient and scalable, making it suitable for large-scale person identification tasks.

In summary, our work presents a comprehensive and efficient solution for person identification based on multimodal analysis using the IEMOCAP dataset. By combining feature extraction and end-to-end learning in an integrated manner, we aim to achieve superior performance in identifying individuals accurately and reliably across different modalities. 
## Paper link and Database

Multimodal End-to-End Sparse Model, by Wenliang Dai *, Samuel Cahyawijaya *, Zihan Liu, Pascale Fung.

Link for the research paper: [https://arxiv.org/pdf/2103.09666.pdf](https://arxiv.org/pdf/2103.09666.pdf)

To know more about the IEMOCAP dataset, click here:
[IEMOCAP](https://sail.usc.edu/iemocap/)


To download the dataset, you can fill the form below for the owner to provide with the database access:

[form link](https://docs.google.com/forms/d/e/1FAIpQLScBecgI2K5bFTrXi_-05IYSSwOcqL5mX7dh57xcJV1m_NoznA/viewform)

Directory Structure:

![image](https://github.com/GANeelima/Multimodal-Person-Identification-/assets/114975668/fa031994-6423-4802-8612-40221fe65703)




## Environment:

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
!python main.py -lr=5e-5 -ep=40 -mod=tav -bs=8 --img-interval=500 --datapath='/content/drive/MyDrive/sample folder preprocess' --dataset='iemocap' --optim='adam' --early-stop=6 --loss=bce --cuda=3 --model=mme2e --num-persons=10 --trans-dim=64 --trans-nlayers=4 --trans-nheads=4 --text-lr-factor=10 --text-model-size=base --text-max-len=100
--fusion='early' --hfc-sizes=[300, 144, 35] --audio-feautre-type=0
```

### Train the sparse MME2E

```console
python main.py -lr=5e-5 -ep=40 -mod=tav -bs=2 --img-interval=500 --datapath='/content/drive/MyDrive/sample folder preprocess' --dataset='iemocap' --optim='adam' --early-stop=6 --loss=bce --cuda=3 --model=mme2e_sparse --num-persons=10 --trans-dim=64 --trans-nlayers=4 --trans-nheads=4 --text-lr-factor=10 -st=0.8 --text-model-size=base --text-max-len=100 --fusion='early' --hfc-sizes=[300, 144, 35] --audio-feautre-type=0
```

For reference, the github link for Multimodal-End to End Sparse Model is given here:
[link](https://github.com/wenliangdai/Multimodal-End2end-Sparse.git/)
