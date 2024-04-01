<p align="center">
    <h1 align="center">Lightning Unet</h1>
</p>

Lihgtning Unet is unet with Lightning Attention for Image Segmentation

## Setup
Environment Requirements
- Python 3.6 or later
- CUDA <= 11.7 (Only Tested On 11.3, On RTX4090, a high CUDA Version will lead to error)
- GPU RAM >= 16GB (For Batch 12)
- Requirements for Lightning Attention

Clone this repo and install python packages

```
git clone 
cd LightningUNet
pip install -r requirements.txt
```

## Download Pretrained Model

Download model from [][]

## Train

### Download Dataset

Download Dataset, and convert them to slices

```
cd tools

python convert_ct.py    # convert Dataset701_AbdomenCT to slices
python convert_mr.py    # convert Dataset701_AbdomenMR to slices
```

### Train

Run the bash Script or run python Script manually

```
bash train.sh

python train.py --cfg configs/m.yaml --dataset ../Dataset701_SlicesCT --epochs 200 --output model --base_lr 0.05 --batch_size 12
```

> [!NOTE]
> The Dataloader can load both volume and slice dataset, However, the train Script can only accept slices, so that we can train the model with batchs

> [!NOTE]
> Due to Linear Attention, the model loss coming down is slow

## Inference

Run the bash Script or run python Script manually

The Inference Script can accept both slices and volumes, for slices, only one slice will be inferenced

```
bash inference.sh

python inference.py --data ../Dataset701_AbdomenCT --output ./prediction --model ./model/pretrained.pth
```

## References

* [LightningAttention](https://github.com/OpenNLPLab/lightning-attention)
* [SwinUnet](https://github.com/HuCaoFighting/Swin-Unet)
* [U-Mamba](https://wanglab.ai/u-mamba.html)