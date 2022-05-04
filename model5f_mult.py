import math

import torch as t
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat


class GDEncoder(nn.Module):
    def __init__(self, args):
        super(GDEncoder, self).__init__()
        self.device = args['device']
        self.lstm_encoder_size = args['lstm_encoder_size']
        self.n_head = args['n_head']
        self.att_out = args['att_out']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.f_length = args['f_length']
        self.relu_param = args['relu']
        self.traj_linear_hidden = args['traj_linear_hidden']
        self.use_maneuvers = args['use_maneuvers']
        self.use_elu = args['use_elu']
        self.use_spatial = args['use_spatial']
        self.dropout = args['dropout']
        # traj encoder
        self.linear1 = nn.Linear(self.f_length, self.traj_linear_hidden)
        self.lstm = nn.LSTM(self.traj_linear_hidden, self.lstm_encoder_size)
        # activation function
        if self.use_elu:
            self.activation = nn.ELU()
        else:
            self.activation = nn.LeakyReLU(self.relu_param)
        #  attention embeding
        self.qff = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.kff = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.vff = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.first_glu = GLU(
            input_size=self.n_head * self.att_out,
            hidden_layer_size=self.lstm_encoder_size,
            dropout_rate=self.dropout)
        self.second_glu = GLU(
            input_size=self.n_head * self.att_out,
            hidden_layer_size=self.lstm_encoder_size,
            dropout_rate=self.dropout)
        #  time attention
        self.qt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.kt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.vt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        #  addAndNorm
        self.addAndNorm = AddAndNorm(self.lstm_encoder_size)
        self.fc = nn.Linear(self.lstm_encoder_size * 2, self.lstm_encoder_size)

    def forward(self, hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls):
        if self.f_length == 5:
            hist = t.cat((hist, cls, va), -1)
            nbrs = t.cat((nbrs, nbrscls, nbrsva), -1)
        elif self.f_length == 6:
            hist = t.cat((hist, cls, va, lane), -1)
            nbrs = t.cat((nbrs, nbrscls, nbrsva, nbrslane), -1)
        # self agent
        hist_enc = self.activation(self.linear1(hist))
        hist_hidden_enc, (_, _) = self.lstm(hist_enc)
        hist_hidden_enc = hist_hidden_enc.permute(1, 0, 2)
        # nbrs agent
        nbrs_enc = self.activation(self.linear1(nbrs))
        nbrs_hidden_enc, (_, _) = self.lstm(nbrs_enc)
        mask = mask.view(mask.size(0), mask.size(1) * mask.size(2), mask.size(3))
        mask = repeat(mask, 'b g s -> t b g s', t=self.in_length)
        # scatter grid embeding
        soc_enc = t.zeros_like(mask).float()
        soc_enc = soc_enc.masked_scatter_(mask, nbrs_hidden_enc)

        query = self.qff(hist_hidden_enc)
        #  度
        _, _, embed_size = query.shape
        query = t.cat(t.split(t.unsqueeze(query, 2), int(embed_size / self.n_head), -1), 1)
        keys = t.cat(t.split(self.kff(soc_enc), int(embed_size / self.n_head), -1), 0).permute(1, 0, 3, 2)
        values = t.cat(t.split(self.vff(soc_enc), int(embed_size / self.n_head), -1), 0).permute(1, 0, 2, 3)
        a = t.matmul(query, keys)
        a /= math.sqrt(self.lstm_encoder_size) 
        a = t.softmax(a, -1)
        values = t.matmul(a, values)
        values = t.cat(t.split(values, int(hist.shape[0]), 1), -1)
        values = values.squeeze(2)
        # gate
        spa_values, _ = self.first_glu(values)

        # Residual connection
        values = self.addAndNorm(hist_hidden_enc, spa_values)
        # temporal attention begin---------------------------
        qt = t.cat(t.split(self.qt(values), int(embed_size / self.n_head), -1), 0)
        kt = t.cat(t.split(self.kt(values), int(embed_size / self.n_head), -1), 0).permute(0, 2, 1)
        vt = t.cat(t.split(self.vt(values), int(embed_size / self.n_head), -1), 0)
        a = t.matmul(qt, kt)
        a /= math.sqrt(self.lstm_encoder_size) 
        a = t.softmax(a, -1)
        values = t.matmul(a, vt)
        values = t.cat(t.split(values, int(hist.shape[1]), 0), -1)
        # ------------------------
        # gate
        time_values, _ = self.second_glu(values)
        # Residual connection,
        if self.use_spatial:
            values = self.addAndNorm(hist_hidden_enc, spa_values, time_values)
        else:
            values = self.addAndNorm(hist_hidden_enc, time_values)
        return values


def outputActivation(x):
    muX = x[:, :, 0:1]
    muY = x[:, :, 1:2]
    sigX = x[:, :, 2:3]
    sigY = x[:, :, 3:4]
    rho = x[:, :, 4:5]
    sigX = t.exp(sigX)
    sigY = t.exp(sigY)
    rho = t.tanh(rho)
    out = t.cat([muX, muY, sigX, sigY, rho], dim=2)
    return out


class AddAndNorm(nn.Module):
    def __init__(self, hidden_layer_size):
        super(AddAndNorm, self).__init__()

        self.normalize = nn.LayerNorm(hidden_layer_size)

    def forward(self, x1, x2, x3=None):
        if x3 is not None:
            x = t.add(t.add(x1, x2), x3)
        else:
            x = t.add(x1, x2)
        return self.normalize(x)


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.relu_param = args['relu']
        self.use_elu = args['use_elu']
        self.use_maneuvers = args['use_maneuvers']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.encoder_size = args['lstm_encoder_size']
        self.n_head = args['n_head']
        self.att_out = args['att_out']
        self.device = args['device']
        self.cat_pred = args['cat_pred']
        self.use_mse = args['use_mse']
        self.lon_length = args['lon_length']
        self.lat_length = args['lat_length']
        if self.use_maneuvers or self.cat_pred:
            self.mu_f = 16
        else:
            self.mu_f = 0
        if self.use_elu:
            self.activation = nn.ELU()
        else:
            self.activation = nn.LeakyReLU(self.relu_param)

        self.lstm = t.nn.LSTM(self.encoder_size, self.encoder_size)
        if self.use_mse:
            self.linear1 = nn.Linear(self.encoder_size, 2)
        else:
            self.linear1 = nn.Linear(self.encoder_size, 5)
        self.lat_linear = nn.Linear(self.lat_length, 8)
        self.lon_linear = nn.Linear(self.lon_length, 8)

        self.dec_linear = nn.Linear(self.encoder_size + self.lat_length + self.lon_length, self.encoder_size)

    def forward(self, dec, lat_enc, lon_enc):

        if self.use_maneuvers or self.cat_pred:
            lat_enc = lat_enc.unsqueeze(1).repeat(1, self.out_length, 1).permute(1, 0, 2)
            lon_enc = lon_enc.unsqueeze(1).repeat(1, self.out_length, 1).permute(1, 0, 2)
            dec = t.cat((dec, lat_enc, lon_enc), -1)
            dec = self.dec_linear(dec)
        h_dec, _ = self.lstm(dec)
        fut_pred = self.linear1(h_dec)
        if self.use_mse:
            return fut_pred
        else:
            return outputActivation(fut_pred)


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.device = args['device']
        self.lstm_encoder_size = args['lstm_encoder_size']
        self.n_head = args['n_head']
        self.att_out = args['att_out']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.f_length = args['f_length']
        self.relu_param = args['relu']
        self.train_flag = args['train_flag']
        self.traj_linear_hidden = args['traj_linear_hidden']
        self.use_maneuvers = args['use_maneuvers']
        self.lat_length = args['lat_length']
        self.lon_length = args['lon_length']
        self.use_elu = args['use_elu']
        self.use_true_man = args['use_true_man']
        self.Decoder = Decoder(args=args)
        self.mu_fc1 = t.nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.mu_fc = t.nn.Linear(self.n_head * self.att_out, self.lstm_encoder_size)
        self.op_lat = t.nn.Linear(self.lstm_encoder_size, self.lat_length)
        self.op_lon = t.nn.Linear(self.lstm_encoder_size, self.lon_length)

        if self.use_elu:
            self.activation = nn.ELU()
        else:
            self.activation = nn.LeakyReLU(self.relu_param)
        self.normalize = nn.LayerNorm(self.lstm_encoder_size)
        #  bcloss
        #  95.56%     3.45%   0.99%
        #  5659826    204302  58739
        #  1          2       3

        self.mapping = t.nn.Parameter(t.Tensor(self.in_length, self.out_length, self.lat_length + self.lon_length))
        nn.init.xavier_uniform_(self.mapping, gain=1.414)  # Glorot init
        self.manmapping = t.nn.Parameter(t.Tensor(self.in_length, 1))
        nn.init.xavier_uniform_(self.mapping, gain=1.414)  # Glorot init

    def forward(self, values, lat_enc, lon_enc):
        # 选择mapping
        maneuver_state = values[:, -1, :]
        maneuver_state = self.activation(self.mu_fc1(maneuver_state))
        maneuver_state = self.activation(self.normalize(self.mu_fc(maneuver_state)))
        lat_pred = F.softmax(self.op_lat(maneuver_state), dim=-1)
        lon_pred = F.softmax(self.op_lon(maneuver_state), dim=-1)
        if self.train_flag:
            if self.use_true_man:
                lat_man = t.argmax(lat_enc, dim=-1).detach()
                lon_man = t.argmax(lon_enc, dim=-1).detach()
            else:
                lat_man = t.argmax(lat_pred, dim=-1).detach().unsqueeze(1)
                lon_man = t.argmax(lon_pred, dim=-1).detach().unsqueeze(1)
                lat_enc_tmp = t.zeros_like(lat_pred)
                lon_enc_tmp = t.zeros_like(lon_pred)
                lat_man = lat_enc_tmp.scatter_(1, lat_man, 1)
                lon_man = lon_enc_tmp.scatter_(1, lon_man, 1)
            index = t.cat((lat_man, lon_man), dim=-1).permute(-1, 0)
            mapping = F.softmax(t.matmul(self.mapping, index).permute(2, 1, 0), dim=-1)
            dec = t.matmul(mapping, values).permute(1, 0, 2)
            if self.use_maneuvers:
                fut_pred = self.Decoder(dec, lat_enc, lon_enc)
                return fut_pred, lat_pred, lon_pred
            else:
                fut_pred = self.Decoder(dec, lat_pred, lon_pred)
                return fut_pred, lat_pred, lon_pred
        else:
            out = []
            for k in range(self.lon_length):
                for l in range(self.lat_length):
                    lat_enc_tmp = t.zeros_like(lat_enc)
                    lon_enc_tmp = t.zeros_like(lon_enc)
                    lat_enc_tmp[:, l] = 1
                    lon_enc_tmp[:, k] = 1
                    index = t.cat((lat_enc_tmp, lon_enc_tmp), dim=-1).permute(-1, 0)
                    mapping = F.softmax(t.matmul(self.mapping, index).permute(2, 1, 0), dim=-1)
                    dec = t.matmul(mapping, values).permute(1, 0, 2)
                    fut_pred = self.Decoder(dec, lat_enc_tmp, lon_enc_tmp)
                    out.append(fut_pred)
            return out, lat_pred, lon_pred


# gate
class GLU(nn.Module):
    # Gated Linear Unit+
    def __init__(self,
                 input_size,
                 hidden_layer_size,
                 dropout_rate=None,
                 ):
        super(GLU, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = nn.Dropout(self.dropout_rate)
        self.activation_layer = t.nn.Linear(input_size, hidden_layer_size)
        self.gated_layer = t.nn.Linear(input_size, hidden_layer_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.dropout_rate is not None:
            x = self.dropout(x)
        activation = self.activation_layer(x)
        gated = self.sigmoid(self.gated_layer(x))
        return t.mul(activation, gated), gated
