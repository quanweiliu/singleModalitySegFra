## Single Modality SemanticSegmentation


这个代码在相同数据集，相同模型的条件下，精度没有 TilewiseSegFra 高。应该是从数据处理 / 数据增强 / 优化器 / lr_schedule 等方面优化。但是没有必要了，直接迭代 TilewiseSegFra 这个库。


此仓库只作为熟悉 [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) 这个库的学习代码。

进一步学习 semantic segmentation refer：[TilewiseSegFra](https://github.com/quanweiliu/TilewiseSegFra)

### 一点小见解

高光谱数据归一化极大的拖慢了速度，并且精度下降了好多了，ref
- 0818-1221-unet-OHEMLoss-resnet18
- 0818-1222-A2FPN-OHEMLoss-resnet18


所以对于高光谱数据就不用归一化了，ref
- 0818-1421-A2FPN-OHEMLoss-resnet18
- 0818-1423-unet-OHEMLoss-resnet18


### Reference: 
https://github.com/jsten07/CNNvsTransformer

https://github.com/suryajayaraman/Semantic-Segmentation-using-Deep-Learning