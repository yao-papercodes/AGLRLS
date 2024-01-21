# Adaptive Global-Local Representation Learning and Selection
Implementation of papers:
- [Adaptive Global-Local Representation Learning and Selection for Cross-Domain Facial Expression Recognition](https://ieeexplore.ieee.org/document/10404024/authors#authors)  
  IEEE Transactions on Multimedia (IEEE TMM), 2024.  
  Yuefang Gao, Yuhao Xie, Zeke Zexi Hu, Tianshui Chen, Liang Lin
## Environment
Ubuntu 22.04.2 LTS, python 3.8.10, PyTorch 1.9.0
## Datasets
Application website： [CK+](http://www.jeffcohn.net/wp-content/uploads/2020/10/2020.10.26_CK-AgreementForm.pdf100.pdf.pdf), [JAFFE](https://zenodo.org/record/3451524#.YXdc1hpBw9E), [SFEW 2.0](https://cs.anu.edu.au/few/AFEW.html), [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data), [ExpW](http://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html), [RAF](http://www.whdeng.cn/raf/model1.html).
## Trained Models
Saved in [here](https://pan.baidu.com/s/1Uhf4XeEFjHd2OgjvMNORnA?pwd=oi5d).
## Usage
```bash
cd code
bash TrainOnSourceDomain.sh     # First step
bash TransferToTargetDomain.sh  # Second step
```
