"""
paper1: Benchmarking Multimodal Sentiment Analysis
paper2: Recognizing Emotions in Video Using Multimodal DNN Feature Fusion
From: https://github.com/rhoposit/MultimodalDNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..subNets import SubNet, TextSubNet

__all__ = ['LF_DNN']

# 后期特征融合模型
"""
先定义不同模态的字网络，然后将原始特征输入到各自的子网络中，得到各自的特征表示，然后将这些特征表示进行拼接，最后通过全连接层进行分类或回归。
不要求三个模态具有相同的序列长度
但拼接本身仍然没有复杂的跨模态交互机制
"""
class LF_DNN(nn.Module):
    """
    late fusion using DNN
    """
    def __init__(self, args):
        super(LF_DNN, self).__init__()
        self.text_in, self.audio_in, self.video_in = args.feature_dims
        self.text_hidden, self.audio_hidden, self.video_hidden = args.hidden_dims

        self.text_out= args.text_out
        self.post_fusion_dim = args.post_fusion_dim

        self.audio_prob, self.video_prob, self.text_prob, self.post_fusion_prob = args.dropouts

        output_dim = args.num_classes if args.train_mode == "classification" else 1

        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear(self.text_out + self.video_hidden + self.audio_hidden, self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, output_dim)


    def forward(self, text_x, audio_x, video_x):
        """
        audio：[B, 1, Da] → [B, Da]
        video：[B, 1, Dv] → [B, Dv]
        """
        audio_x = audio_x.squeeze(1)
        video_x = video_x.squeeze(1)

        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        text_h = self.text_subnet(text_x)

        fusion_h = torch.cat([audio_h, video_h, text_h], dim=-1)
        x = self.post_fusion_dropout(fusion_h)
        x = F.relu(self.post_fusion_layer_1(x), inplace=True)
        x = F.relu(self.post_fusion_layer_2(x), inplace=True)
        output = self.post_fusion_layer_3(x)

        res = {
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': fusion_h,
            'M': output
        }
        return res
        