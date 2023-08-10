import torch
import torchaudio
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self,
                 eeg_feature_dim,
                 audio_feature_dim,
                 fusion_type,
                 hidden_dim,
                 n_class=2):
        super(FusionModel, self).__init__()
        self.fusion_type = fusion_type
        # self.audio_frontend = AudioFrontEnd(sampling_rate=16000, n_mfcc=audio_feature_dim)
        print(self.fusion_type)
        if self.fusion_type == "intra":
            self.eeg_proj = MLP(eeg_feature_dim, hidden_dim)
            self.audio_proj = MLP(audio_feature_dim, hidden_dim)
            self.classifier = nn.Linear(hidden_dim*2, n_class)
        elif self.fusion_type == "inter":
            self.fusion_dim = eeg_feature_dim + audio_feature_dim
            self.fusion_proj = MLP(self.fusion_dim, hidden_dim)
            self.classifier = nn.Linear(hidden_dim, n_class)
        elif self.fusion_type == "intra_inter":
            self.eeg_proj = MLP(eeg_feature_dim, hidden_dim)
            self.audio_proj = MLP(audio_feature_dim, hidden_dim)
            self.fusion_dim = eeg_feature_dim + audio_feature_dim
            self.fusion_proj = MLP(self.fusion_dim, hidden_dim)
            self.classifier = nn.Linear(hidden_dim*3, n_class)
        
    def forward(self, eeg, audio):
        # mfcc = self.audio_frontend(audio)
        # audio = mfcc.mean(dim=-1)
        if self.fusion_type == "intra":
            h_eeg = self.eeg_proj(eeg.squeeze(1))
            h_audio = self.audio_proj(audio.squeeze(1))
            h_x = torch.cat([h_eeg, h_audio], dim=-1)
            x = self.classifier(h_x)
        elif self.fusion_type == "inter":
            fusion_x = torch.cat([eeg, audio], dim=-1)
            h_fusion = self.fusion_proj(fusion_x.squeeze(1))
            x = self.classifier(h_fusion)
        elif self.fusion_type == "intra_inter":
            fusion_x = torch.cat([eeg, audio], dim=-1)
            h_eeg = self.eeg_proj(eeg.squeeze(1))
            h_audio = self.audio_proj(audio.squeeze(1))
            h_fusion = self.fusion_proj(fusion_x.squeeze(1))
            h_x = torch.cat([h_eeg, h_audio, h_fusion], dim=-1)
            x = self.classifier(h_x)
        return x

    def extractor(self, eeg, audio):
        if self.fusion_type == "intra":
            h_eeg = self.eeg_proj(eeg.squeeze(1))
            h_audio = self.audio_proj(audio.squeeze(1))
            h_x = torch.cat([h_eeg, h_audio], dim=-1)
        elif self.fusion_type == "inter":
            fusion_x = torch.cat([eeg, audio], dim=-1)
            h_x = self.fusion_proj(fusion_x.squeeze(1))
        elif self.fusion_type == "intra_inter":
            h_eeg = self.eeg_proj(eeg.squeeze(1))
            h_audio = self.audio_proj(audio.squeeze(1))
            fusion_x = torch.cat([eeg, audio], dim=-1)
            h_fusion = self.fusion_proj(fusion_x.squeeze(1))
            h_x = torch.cat([h_eeg, h_audio, h_fusion], dim=-1)
        return h_x

class AudioFrontEnd(nn.Module):
    def __init__(self, sampling_rate, n_mfcc):
        super(AudioFrontEnd, self).__init__()
        self.n_fft = int(0.5 * sampling_rate)
        self.win_length = int(0.5 * sampling_rate)
        self.hop_length = int(0.25 * sampling_rate)
        self.n_mels = 96
        self.n_mfcc = n_mfcc
        melkwargs= {
            'n_fft': self.n_fft,
            'n_mels': self.n_mels,
            'win_length': self.win_length,
            'hop_length': self.hop_length,
            }
        self.mfcc_fn = torchaudio.transforms.MFCC(n_mfcc = self.n_mfcc, melkwargs = melkwargs )
    def forward(self, x):
        return self.mfcc_fn(x)


class MLP(nn.Module):
    def __init__(self, in_channel, hidden_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_channel, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        ) 
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        ) 
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        ) 

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

class Res2DMaxPoolModule(nn.Module):
    def __init__(self, input_channels, output_channels, pooling=2):
        super(Res2DMaxPoolModule, self).__init__()
        self.conv_1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)
        # residual
        self.diff = False
        if input_channels != output_channels:
            self.conv_3 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
            self.bn_3 = nn.BatchNorm2d(output_channels)
            self.diff = True

    def forward(self, x):
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        out = self.mp(self.relu(out))
        return out
    
class ResNet(nn.Module):
    def __init__(self, in_channel, conv_ndim, n_class=2):
        super(ResNet, self).__init__()
        self.conv_ndim = conv_ndim
        self.layer1 = Res2DMaxPoolModule(in_channel, conv_ndim, pooling=(4, 4))
        self.layer2 = Res2DMaxPoolModule(conv_ndim, conv_ndim, pooling=(4, 4))
        self.layer3 = Res2DMaxPoolModule(conv_ndim, conv_ndim, pooling=(2, 2))
        self.layer4 = Res2DMaxPoolModule(conv_ndim, conv_ndim, pooling=(2, 2))
        self.fc = nn.Linear(conv_ndim, n_class)
    def forward(self, x):
        # CNN
        b, c, t, f = x.shape
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(b, self.conv_ndim)
        out = self.fc(out)
        return out

class Conv_1d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_1d, self).__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(pooling)
    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out

class SampleCNN(nn.Module):
    def __init__(self,
                 n_class=4):
        super(SampleCNN, self).__init__()
        self.layer1 = Conv_1d(32, 128, shape=3, stride=3, pooling=1)
        self.layer2 = Conv_1d(128, 128, shape=3, stride=1, pooling=3)
        self.layer3 = Conv_1d(128, 128, shape=3, stride=1, pooling=3)
        self.layer4 = Conv_1d(128, 256, shape=3, stride=1, pooling=3)
        self.layer5 = Conv_1d(256, 256, shape=3, stride=1, pooling=3)
        self.layer6 = Conv_1d(256, 256, shape=3, stride=1, pooling=3)
        self.layer7 = Conv_1d(256, 512, shape=3, stride=1, pooling=3)
        self.layer8 = Conv_1d(512, 512, shape=3, stride=1, pooling=3)
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(512, n_class)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.dense(x)
        return x