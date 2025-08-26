# ***CADS***

This repo is the implementation of our IEEE Journal of Biomedical and Health Informatics paper entitled [CADS: Causal Inference for Dissecting Essential Genes to Predict Drug Synergy](https://www.google.com/).
In this study, we designed a novel deep learning model ***CADS*** (Causal Adjustment for Drug Synergy) based on causal inference for the prediction of drug synergy. Then we integrated ***CADS*** into several carefully selected base models and demonstrated its efficient predictive performance through comprehensive experiments. Finally, we conducted comprehensive experiments to validate the ***CADS***'s performance and robustness, and conducted multi-granularity analyses to identify and interpret causal genes associated with drug synergy.

## Framework Overview

<img src="modelFramework.png" width="1000px">

The pipeline of CADS. The ***CADS*** framework primarily consists of three main components: **(a)** *Drug Feature Extraction* is employed to extract drug features from sequences or 2D molecular graphs; **(b)** *Causal Mask Encoder* generates learnable causal mask pairs to decouple causal information from trivial information leveraging cell line profiles; **(c-d)** *Drug-Cell Fusion* employs Multi-Layer Perceptrons (MLP) or Cross-Attention (CA) and Multi-head Self-Attention (MSA) to fuse drug features with decoupled cell representations; **(e)** *Causal Learning Module* learns hard-to-observe causal information by training the causal and trivial components separately with ground truth and fake labels, respectively.

## Step-by-Step Guide for Running ***CADS***

### Clone the Repository

```bash
git clone https://github.com/HuaiwuZhang/causalDC.git
cd causalDC
```

### Create and Activate a New Environment

```bash
conda create -n causalDC python=3.9 -y
conda activate causalDC
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### (Optional) Weights & Biases

If you want to enable [Weights & Biases]() logging:

```bash
wandb login
```

Add your wandb key in main.py

```python
wandb.login(key="<your-wandb-key>")
```

### Model Training & Testing

**Example 1**: Training *SynergyX* as a base model from scratch with ***CADS*** embeddings, using *Loewe* as the drug synergy metric.
* 
```
python main.py --model synergyx_causal --labels loewe
```
(Here we added functionality to save and load training parameters so you can start or stop training at any time.)


**Example 2**: Few-shot learning using *DeepSynergy* as a base model from scratch with ***CADS*** embeddings, based on the *s-score* metric, saving 70% of the training data.

```
python main.py --model deepsynergy_causal --torch_seed 40 --n_exp 1 --labels sscore --few_shot_ratio 0.7
```

**Example 3**: Calculating the evaluation metrics for the valid set using *DeepDDS* as a base model from scratch with ***CADS*** embeddings, based on *ZIP* metric.

```
python main.py --model synergyx_causal --labels zip --test_valid True --pretrain False --testing False
```


### More Function

To test more function based on above-trained models, you can use the following command:

**Example 1**: Obtaining the cell embedding, using *SynergyX* as a base model from scratch with ***CADS*** embeddings, based on *S-Score* metric.

```
python main.py --wandb False --model synergyx_causal --get_cell_embedding True --labels sscore --pretrain False --testing False
```

**Example 2**: Evaluation of novel drug combinations.

```
python main.py --model synergyx_causal --labels sscore --pretrain False --testing False --wandb False --novel True
```

**Example 3**: Retrieving general causal genes across cell lines.

```
python main.py --model synergyx_causal --infer True --labels sscore --pretrain False
```

**Example 4**: Retrieving causal genes that affect specific drug combinations.

```
python main.py --model synergyx_causal --wandb False --pretrain False --testing False --labels sscore --IG_drugA Ruxolitinib --IG_drugB Vismodegib
python main.py --model synergyx_causal --wandb False --pretrain False --testing False --labels sscore --IG_drugA Crizotinib --IG_drugB Lenalidomide
```

## Folder Structure

We list the code of the major modules as follows:

1. The main function to train/test our model:  [click here.](experiments/DeepPA/main.py "1")
2. The source code of base models and causal models: [click here.](models "2")
3. The trainer/tester: [click here.](src/trainers/deeppa_trainer.py "3")
4. Data preparation and preprocessing are located at [click here.](experiments/DeepPA/main.py "4")
5. Computations: [click here.](src/utils "5")

## Arguments

We introduce some major arguments of our main function here.

Training settings:

- mode: indicating the mode (train or test).
- n_exp: experimental group number.
- gpu: which gpu used to train.
- seed: the random seed for experiments. (default: 0)
- dataset: dataset path for the experiment.
- batch_size: batch size of training or testing.
- seq_len: the length of historical steps.
- horizon: the length of future steps.
- input_dim: the dimension of inputs.
- output_dim: the dimension of inputs.
- max_epochs: maximum number of training epochs.
- patience: the patience of early stopping.
- save_preds: whether to save prediction results.
- wandb: whether to use wandb.

Model hyperparameters:

- dropout: dropout rate.
- n_blocks: number of layers of SLBlock and TLBlock.
- n_hidden: hidden dimensions in SLBlock and TLBlock.
- n_heads: number of heads in MSA.
- spatial_flag: whether to use SLBlock.
- temporal_flag: whether to use TLBlock.
- spatial_encoding: whether to treat temporal factor as a station.
- temporal_encoding: Whether to incorporate spatial factor into TLBlock.
- temporal_PE: whether to use temporal position encoding.
- GCO: whether to use GCO.
- GCO_Thre: the proportion of low frequency signals.
- base_lr: base learning rate.
- lr_decay_ratio: learning rate decay ratio.

## License

The <b>SINPA</b> dataset is released under the Singapore Open Data Licence: [https://beta.data.gov.sg/open-data-license](https://beta.data.gov.sg/open-data-license).

## Citation

If you find our work useful in your research, please cite:

```
@inproceedings{zhang2024predicting,
  title={Predicting Parking Availability in Singapore with Cross-Domain Data: A New Dataset and A Data-Driven Approach},
  author={Zhang, Huaiwu and Xia, Yutong and Zhong, Siru and Wang, Kun and Tong, Zekun and Wen, Qingsong and Zimmermann, Roger and Liang, Yuxuan},
  booktitle={Proceedings of the Thirty-third International Joint Conference on Artificial Intelligence, IJCAI-24},
  year={2024}
}
```
