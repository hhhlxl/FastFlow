import FrEIA.framework as Ff
import FrEIA.modules as Fm
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import constants as const


def subnet_conv_func(kernel_size, hidden_ratio):
    def subnet_conv(in_channels, out_channels):
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
        )


    return subnet_conv


def nf_fast_flow(input_chw, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
    # 一个node即为一个INN块,INN块就是一个仿射结构
    nodes = Ff.SequenceINN(*input_chw)
    for i in range(flow_steps):
        if i % 2 == 1 and not conv3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            # 这个参数用于防止指数爆炸
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes


class FastFlow(nn.Module):
    def __init__(
        self,
        backbone_name,
        # 这里给ResNet18设置flow_steps=8
        flow_steps,
        input_size,
        conv3x3_only=False,
        hidden_ratio=1.0,
    ):
        super(FastFlow, self).__init__()
        assert (
            backbone_name in const.SUPPORTED_BACKBONES
        ), "backbone_name must be one of {}".format(const.SUPPORTED_BACKBONES)

        # 如果骨干网络是CaiT或者DeiT，channel=768，scale=16，这两个是啥？
        if backbone_name in [const.BACKBONE_CAIT, const.BACKBONE_DEIT]:
            self.feature_extractor = timm.create_model(backbone_name, pretrained=True)
            channels = [768]
            scales = [16]
        else:
            self.feature_extractor = timm.create_model(
                backbone_name,
                pretrained=True,
                features_only=True,
                out_indices=[1, 2, 3],
            )
            # 如果骨干网络是resnet，channle=feature_info的channel数，scales=feature_info的reduction，这两个是啥啊
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()

            # for transformers, use their pretrained norm w/o grad
            # for resnets, self.norms are trainable LayerNorm
            self.norms = nn.ModuleList()
            for in_channels, scale in zip(channels, scales):
                self.norms.append(
                    nn.LayerNorm(
                        [in_channels, int(input_size / scale), int(input_size / scale)],
                        elementwise_affine=True,
                    )
                )

        # 这是冻结模型吗
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.nf_flows = nn.ModuleList()
        # channels是通道数，scales是什么？
        for in_channels, scale in zip(channels, scales):
            self.nf_flows.append(
                nf_fast_flow(
                    [in_channels, int(input_size / scale), int(input_size / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                )
            )
        self.input_size = input_size

    def forward(self, x):
        # 输入：batch_size*3*256*256，batch_size张3通道256*256的图像
        self.feature_extractor.eval()
        # 检验feature_extractor是否为ViT
        if isinstance(
            self.feature_extractor, timm.models.vision_transformer.VisionTransformer
        ):
            # 将输入图像x分为一系列的小补丁（patches）
            x = self.feature_extractor.patch_embed(x)
            # 创建一个分类令牌（cls_token）的扩展版本，其形状与 x 的批次大小相匹配
            cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)
            # 如果ViT没有蒸馏token，将分类令牌（cls_token）与图像补丁（x）沿着第二个维度（dim=1）连接起来
            if self.feature_extractor.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            # 如果有蒸馏token，则需要将分类令牌、蒸馏令牌和图像补丁都连接起来
            else:
                x = torch.cat(
                    (
                        cls_token,
                        self.feature_extractor.dist_token.expand(x.shape[0], -1, -1),
                        x,
                    ),
                    dim=1,
                )
            # 将位置嵌入(pos embed)添加到x上，并为结果应用位置丢弃(pos drop)，将位置信息引入到补丁嵌入中。
            x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
            # 遍历前8个Transformer块,将x传递给每个块
            for i in range(8):  # paper Table 6. Block Index = 7
                x = self.feature_extractor.blocks[i](x)
            # 对x进行归一化
            x = self.feature_extractor.norm(x)
            # 删除前两个维度（即分类令牌和可能的蒸馏令牌）
            x = x[:, 2:, :]
            # 获取x的形状，批次大小、补丁数、通道数
            N, _, C = x.shape
            # 重新排列张量，本来是0（批次数），1（补丁数），2（通道数），现在把2换到1的位置，1换到2的位置
            x = x.permute(0, 2, 1)
            # 将x的形状变为 [N, C, H, W]，H和W=输入图像大小除以16
            x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
            # 将重塑后的x存在features
            features = [x]
        elif isinstance(self.feature_extractor, timm.models.cait.Cait):
            x = self.feature_extractor.patch_embed(x)
            x = x + self.feature_extractor.pos_embed
            x = self.feature_extractor.pos_drop(x)
            for i in range(41):  # paper Table 6. Block Index = 40
                x = self.feature_extractor.blocks[i](x)
            N, _, C = x.shape
            x = self.feature_extractor.norm(x)
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
            features = [x]
        # 如果用resnet提取特征,对提取的每一层的特征进行归一化
        else:
            features = self.feature_extractor(x)
            # 提取的三层特征，[64, 64, 64] [128, 32, 32] [256, 16, 16]
            # print('features:', len(features))
            features = [self.norms[i](feature) for i, feature in enumerate(features)]

        loss = 0
        outputs = []
        # 对三层特征的每一层进行处理，分别送入flow网络，分别得到输出、雅各比行列式并计算损失，最后把三次的损失加起来返回
        for i, feature in enumerate(features):
            # output是经过变换后的特征,log_jac_dets是对数雅各比行列式
            output, log_jac_dets = self.nf_flows[i](feature)
            # 为什么要mean一下？
            loss += torch.mean(
                0.5 * torch.sum(output**2, dim=(1, 2, 3)) - log_jac_dets
            )
            outputs.append(output)
        ret = {"loss": loss}

        # 推理时
        if not self.training:
            anomaly_map_list = []
            # 又分别对三层特征的输出进行处理
            for output in outputs:
                # 计算log probability,计算output的平方，然后在第二个维度（dim=1）上取均值，并保持这个维度（keepdim=True），最后乘以-0.5
                # 这就是anomaly score？shape=[batch_size, 1, H, W]，后面的分别是32和16
                # 得把这里加上image-level的score计算，求均值？因为下面的也是求均值，我猜。。二维的怎么求均值啊，服了
                # 为什么又要mean?
                log_prob = -torch.mean(output**2, dim=1, keepdim=True) * 0.5
                # print('log_prob: ', log_prob.shape)
                # 将上一步得到的log_prob转换为概率
                prob = torch.exp(log_prob)
                # 对 -prob 进行上采样为输入图像大小
                a_map = F.interpolate(
                    -prob,
                    size=[self.input_size, self.input_size],
                    mode="bilinear",
                    align_corners=False,
                )
                # a_map的尺寸：[batch_size, 1, 256, 256]
                anomaly_map_list.append(a_map)
            # 异常映射列表本来是list，现在在-1维度把三个a_map堆叠起来，变成[batch_size, 1, 256, 256, 3]
            anomaly_map_list = torch.stack(anomaly_map_list, dim=-1)
            # 在-1维度（即堆叠的维度）上计算anomaly_map_list的平均值，得到一个单一的异常映射anomaly_map
            anomaly_map = torch.mean(anomaly_map_list, dim=-1)
            # 经过上面的操作，anomaly_map变成了和输入长宽一样的映射图，尺寸为[batch_size, 1, 256, 256]，通道数为1（跟mask一样）
            ret["anomaly_map"] = anomaly_map

        return ret
