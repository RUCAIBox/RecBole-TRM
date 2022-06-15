# RecBole-TRM

**RecBole-TRM** is a library built upon [PyTorch](https://pytorch.org) and [RecBole](https://github.com/RUCAIBox/RecBole) for reproducing and developing recommendation algorithms based on Transformers (TRMs). Our library includes algorithms covering two major categories:
* **Sequential Recommendation**: TiSASRec, SSE-PT, LightSANs, gMLP, CORE
* **News Recommendation**: NRMS, NAML, NPA

## Highlights

* **Easy-to-use API**:
    Our library shares unified API and input (atomic files) as RecBole.
* **Fair reproducibility and comparison**:
    Our library provides fair reproducibility and comparison in a systematic mechanism.
* **Extensive Transformer library**:
    Our library provides extensive API based on common Transformer layers, one can further develop new models easily based on our library.

## Requirements

```
recbole>=1.0.0
pyg>=2.0.4
pytorch>=1.7.0
python>=3.7.0
```

## Quick-Start

With the source code, you can use the provided script for initial usage of our library:

```bash
python run_recbole_trm.py
```

If you want to change the models or datasets, just run the script by setting additional command parameters:

```bash
python run_recbole_trm.py -m [model] -d [dataset]
```

## Implemented Models

We list currently supported models according to category:

**Sequential Recommendation**:

* **[TiSASRec](/recbole/model/transformer_recommender/tisasrec.py)** from Li *et al.*: [Time Interval Aware Self-Attention for Sequential Recommendation](https://dl.acm.org/doi/10.1145/3336191.3371786) (WSDM 2020).
* **[SSE-PT](/recbole/model/transformer_recommender/ssept.py)** from Wu *et al.*: [SSE-PT: Sequential Recommendation Via Personalized Transformer](https://dl.acm.org/doi/10.1145/3383313.3412258) (RecSys 2020).
* **[LightSANs](/recbole/model/sequential_recommender/lightsans.py)** from Fan *et al.*: [Lighter and Better: Low-Rank Decomposed Self-Attention Networks for Next-Item Recommendation](https://dl.acm.org/doi/10.1145/3404835.3462978) (SIGIR 2021).
* **[gMLP](/recbole/model/transformer_recommender/gmlp.py)** from Liu *et al.*: [Pay Attention to MLPs](https://arxiv.org/abs/2105.08050) (NeurIPS 2021).
* **[CORE](/recbole/model/sequential_recommender/core.py)** from Hou *et al.*: [CORE: Simple and Effective Session-based Recommendation within Consistent Representation Space](https://arxiv.org/abs/2204.11067) (SIGIR 2022).

**News Recommendation**:

* [NRMS](https://aclanthology.org/D19-1671/) from Wu *et al.*: [Neural News Recommendation with Multi-Head Self-Attention](https://aclanthology.org/D19-1671/) (EMNLP/IJCNLP 2019).
* [NAML](https://arxiv.org/abs/1907.05576) from Wu *et al.*: [Neural News Recommendation with Attentive Multi-View Learning](https://arxiv.org/abs/1907.05576) (IJCAI 2019).
* [NPA](https://arxiv.org/abs/1907.05559) from Wu *et al.*: [NPA: Neural News Recommendation with Personalized Attention](https://arxiv.org/abs/1907.05559) (KDD 2019).

## Experiments

For more details about experiments including the hyper-parameters of the implemented models, you can refer to [[link]](experiments/README.md).

## The Team

RecBole-TRM is developed and maintained by members from [RUCAIBox](http://aibox.ruc.edu.cn/), the main developers are Wenqi Sun ([@wenqisun](https://github.com/wenqisun)) and Xinyan Fan ([@BELIEVEfxy](https://github.com/BELIEVEfxy)).

## Acknowledgement

The implementation is based on the open-source recommendation library [RecBole](https://github.com/RUCAIBox/RecBole).

Please cite the following paper as the reference if you use our code or processed datasets.

```
@inproceedings{zhao2021recbole,
  title={Recbole: Towards a unified, comprehensive and efficient framework for recommendation algorithms},
  author={Wayne Xin Zhao and Shanlei Mu and Yupeng Hou and Zihan Lin and Kaiyuan Li and Yushuo Chen and Yujie Lu and Hui Wang and Changxin Tian and Xingyu Pan and Yingqian Min and Zhichao Feng and Xinyan Fan and Xu Chen and Pengfei Wang and Wendi Ji and Yaliang Li and Xiaoling Wang and Ji-Rong Wen},
  booktitle={{CIKM}},
  year={2021}
}
