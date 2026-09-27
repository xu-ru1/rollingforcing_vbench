# R6B：126-latent Front/U-shape 成本曲线

R6B 继承R6A的126-latent、seed0、warmup+cat/rainy-car/robot、501帧解码检查和
质量盲条件。它重新运行Fixed 0.26/0.40两个锚点，并扫描Front/U-shape候选的真实
diffusion latency，不读取质量结果，不自动冻结最终方法。

## Case

- Vanilla：1个基线case。
- Fixed：0.26 slow anchor、0.40 fast anchor。
- Front：0.28、0.34、0.40、0.46。
- U-shape：0.38、0.44、0.50、0.56。

共11个case，每个case为4个126-latent视频，其中第0条只作warmup，后3条用于
初步时延曲线。所有方法使用同一prompt顺序、seed0和随机轨迹hash。

## 选择规则

R6B只输出候选曲线。slow/fast匹配优先使用平均diffusion latency，目标与对应
Fixed锚点的相对误差不超过2%；随后检查DiT PFLOPs、实际token成本、复用mask和
CV。Front-slow匹配Fixed-slow，Front-fast和U-shape-fast匹配Fixed-fast。

若候选CV超过2%，该点只作为初筛结果；最终配置需要在R6C中按固定prompt重复计时。
若多个阈值产生同一复用mask，保留更快且较低复杂度的点，删除平台重复点。
