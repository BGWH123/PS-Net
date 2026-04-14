import torch
import torch.nn as nn


class PeriodExpert(nn.Module):
    def __init__(self, s, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(s, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, s)

    def forward(self, v):
        # v: (b, c, s)
        v = self.input_proj(v)  # (b, c, d_model)
        v = self.encoder(v)  # (b, c, d_model)
        v = self.output_proj(v)  # (b, c, s)
        return v


class AdaptiveMoE(nn.Module):
    def __init__(self, enc_in, seq_len, num_experts=3, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts

        # gating network
        self.gate = nn.Sequential(
            nn.Linear(enc_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_experts)
        )

        # experts
        self.experts = nn.ModuleList([
            PeriodExpert(s=seq_len, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x, v_list):
        # x: (b, c)
        gate_logits = self.gate(x)  # (b, num_experts)
        gate_weight = torch.softmax(gate_logits, dim=-1)  # (b, num_experts)

        expert_out = []
        for i, v in enumerate(v_list):
            v_out = self.experts[i](v)  # (b, c, s)
            expert_out.append(v_out)

        expert_out = torch.stack(expert_out, dim=1)  # (b, num_experts, c, s)

        gate_weight = gate_weight.unsqueeze(-1).unsqueeze(-1)  # (b, num_experts, 1, 1)
        fused = (gate_weight * expert_out).sum(dim=1)  # (b, c, s)

        return fused, gate_weight


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.use_revin = configs.use_revin

        self.use_period_matrix = True
        self.use_channel_attention = True
        self.medium_cycle = self.cycle_len // 2
        self.short_cycle = self.cycle_len // 4

        if self.use_period_matrix:
            self.period_matrix = nn.Parameter(torch.zeros(self.cycle_len, self.enc_in), requires_grad=True)
            self.period_matrix_medium = nn.Parameter(torch.zeros(self.medium_cycle, self.enc_in), requires_grad=True)
            self.period_matrix_short = nn.Parameter(torch.zeros(self.short_cycle, self.enc_in), requires_grad=True)

        if self.use_channel_attention:
            self.channel_attention = nn.MultiheadAttention(embed_dim=self.seq_len, num_heads=4, batch_first=True,
                                                           dropout=0.5)
            self.channel_attention_medium = nn.MultiheadAttention(embed_dim=self.seq_len, num_heads=4, batch_first=True,
                                                                  dropout=0.5)
            self.channel_attention_short = nn.MultiheadAttention(embed_dim=self.seq_len, num_heads=4, batch_first=True,
                                                                 dropout=0.5)

        self.input_proj = nn.Linear(self.seq_len, self.d_model)
        self.steering_proj = nn.Linear(self.enc_in, self.enc_in)
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )

        # Adaptive Mixture-of-Experts
        self.moe = AdaptiveMoE(
            enc_in=self.enc_in,
            seq_len=self.seq_len,
            num_experts=3,
            d_model=128,
            nhead=4,
            num_layers=2,
            dropout=0.1
        )

        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len)
        )

    def forward(self, x, cycle_index):
        # x: (b, s, c)

        # 1) ReVIN normalization
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # 2) b,s,c -> b,c,s
        x_input = x.permute(0, 2, 1)

        # 3) period matrix retrieval + channel-aware attention
        if self.use_period_matrix:
            gather_index = (cycle_index.view(-1, 1) + torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)) % self.cycle_len
            period_query_long = self.period_matrix[gather_index].permute(0, 2, 1)

            gather_index_medium = (cycle_index.view(-1, 1) + torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)) % self.medium_cycle
            period_query_medium = self.period_matrix_medium[gather_index_medium].permute(0, 2, 1)

            gather_index_short = (cycle_index.view(-1, 1) + torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)) % self.short_cycle
            period_query_short = self.period_matrix_short[gather_index_short].permute(0, 2, 1)

            if self.use_channel_attention:
                steering_vector_long = self.channel_attention(query=period_query_long, key=x_input, value=x_input)[0]
                steering_vector_medium = self.channel_attention_medium(query=period_query_medium, key=x_input, value=x_input)[0]
                steering_vector_short = self.channel_attention_short(query=period_query_short, key=x_input, value=x_input)[0]
            else:
                steering_vector_long = period_query_long
                steering_vector_medium = period_query_medium
                steering_vector_short = period_query_short
        else:
            if self.use_channel_attention:
                steering_vector_long = self.channel_attention(query=x_input, key=x_input, value=x_input)[0]
                steering_vector_medium = self.channel_attention_medium(query=x_input, key=x_input, value=x_input)[0]
                steering_vector_short = self.channel_attention_short(query=x_input, key=x_input, value=x_input)[0]
            else:
                steering_vector_long = 0
                steering_vector_medium = 0
                steering_vector_short = 0

        # 4) Adaptive MoE
        q = x_input.mean(dim=2)
        q = torch.relu(self.steering_proj(q))

        v_list = [steering_vector_long, steering_vector_medium, steering_vector_short]
        fused, gate_weight = self.moe(q, v_list)  # fused: (b, c, s)

        # 5) fuse back and project
        projected = self.input_proj(x_input + fused)

        # 6) MLP
        hidden = self.mlp(projected)

        # 7) output projection
        output = self.output_proj(hidden + projected).permute(0, 2, 1)

        # 8) denormalization
        if self.use_revin:
            output = output * torch.sqrt(seq_var) + seq_mean

        return output
