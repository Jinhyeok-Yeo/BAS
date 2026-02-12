import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
import numpy as np
import time
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import copy

# =======================
# Utils
# =======================
# [BAS 주석] 로그 관리 설정: 장기간 운전되는 공정 모니터링 시스템 특성상
# 날짜별 로그 회전(Rotating) 및 포맷팅은 필수적입니다.
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")

logger = logging.getLogger("myapp")
logger.setLevel(logging.INFO)   # 필요시 DEBUG로 변경
logger.propagate = False        # 루트로 중복 전파 방지

file_handler = TimedRotatingFileHandler(
    LOG_FILE,
    when="midnight",      # 매일 자정 기준
    interval=1,           # 1일마다
    backupCount=7,        # 지난 파일 7개 보관 (일주일)
    encoding="utf-8",
    utc=False,            # KST 기준(로컬 시간). UTC 쓰려면 True
    delay=True            # 첫 로그가 찍히기 전까지 파일을 만들지 않음 (장기 실행시 권장)
)

file_handler.suffix = "%Y-%m-%d"
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logger.handlers.clear()
# logger.addHandler(console)
logger.addHandler(file_handler)


def _slice_future(arr: np.ndarray, t0: int, H: int):
    """
    [BAS 주석] 미래 예측 윈도우 슬라이싱 (Rollout Horizon)
    화학 공정 제어(MPC 등)를 위해서는 현재 시점(t0) 이후 H step 동안의
    미래 상태 변화를 예측해야 합니다.
    - arr: 전체 시계열 데이터
    - t0: 현재 시점
    - H: 예측할 미래 구간 길이 (Horizon)
    """
    # arr: (N,) 또는 (N,F). 항상 미래(t0+1..t0+H)만 자름
    return torch.from_numpy(arr[t0+1 : t0+1+H].astype(np.float32))

def build_feedback_map(인풋변수s: dict, 타깃변수s: dict):
    """
    [BAS 주석] 공정 피드백 루프(Feedback Loop) 매핑 생성
    화학 공정은 연속적입니다. t 시점의 출력(Output)이 t+1 시점의 입력(Input)이 되거나,
    상류 공정의 결과가 하류 공정으로 전파됩니다.

    이 함수는 '예측된 값(y_hat)'을 다음 스텝의 '입력 특성(feature)'으로
    재투입(Re-injection)하기 위한 인덱스 지도를 만듭니다.
    Autoregressive(자기회귀) 시뮬레이션을 위해 필수적입니다.

    스테이지 순서 고정 (물리적 공정 흐름):
      0:'1단 온도_1' (Reactor Zone 1)
      1:'3단 온도_1' (Reactor Zone 3)
      2:'Analyzer'   (Gas Analyzer)
      3:'TMS'        (Tele-Monitoring System, Stack)
    """
    stage_order = ['1단 온도_1','3단 온도_1','Analyzer','TMS']
    stage_index = {name:i for i,name in enumerate(stage_order)}

    # feats_map[stage][stream] = { feature_name: feature_index }
    feats_map = {}
    for s_name, stream_lists in 인풋변수s.items():
        s_idx = stage_index[s_name]
        feats_map[s_idx] = []
        for feat_list in stream_lists:
            feats_map[s_idx].append({fn: k for k, fn in enumerate(feat_list)})

    feedback_map = {0:[], 1:[], 2:[], 3:[]}

    # 각 소스(오너) 스테이지
    for src_name, tgt_list in 타깃변수s.items():
        src_idx = stage_index[src_name]

        for y_k, diff_name in enumerate(tgt_list):
            assert diff_name.endswith('_diff'), f"타깃 '{diff_name}'는 '_diff'로 끝나야 합니다."
            base = diff_name[:-5]  # '_diff' 제거 (변분 예측을 통해 안정성 확보)

            # prev_ref: 오너 스테이지에서 <base>+'_1min'을 찾아야 함 (필수)
            # [BAS 설명] 미분값(diff)을 적분하여 원본 값(Level)을 복원하기 위한 기준점(t-1) 찾기
            prev_ref = None
            for js, fmap in enumerate(feats_map[src_idx]):
                key = base + '_1min'
                if key in fmap:
                    prev_ref = {"stage": src_idx, "stream": js, "feat_idx": fmap[key]}
                    break
            if prev_ref is None:
                # 못 찾으면 스킵(또는 예외 처리)
                continue

            # 모든 타깃 스테이지/스트림을 훑어서 기록 대상 피처 탐색
            # [BAS 설명] 예측된 값이 다른 공정 단계의 입력으로 쓰이는 곳을 찾아서 연결
            for tgt_idx in range(len(stage_order)):
                for j, fmap in enumerate(feats_map[tgt_idx]):
                    # 1) 현재값 컬럼이 있으면: 현재값 로직
                    if base in fmap:
                        feedback_map[src_idx].append({
                            "type": "diff_to_level",
                            "stage": tgt_idx,
                            "stream": j,
                            "feat_idx": fmap[base],
                            "target_is_lag": False,   # 현재값 로직
                            "y_idx": y_k,
                            "prev_ref": prev_ref
                        })
                    # 2) '_1min' 컬럼이 있으면: lag 로직 (현재값 유무와 무관)
                    # [BAS 설명] Lag Feature 업데이트 (시계열 모델의 자기회귀 특성 반영)
                    if base + '_1min' in fmap:
                        feedback_map[src_idx].append({
                            "type": "diff_to_level",
                            "stage": tgt_idx,
                            "stream": j,
                            "feat_idx": fmap[base + '_1min'],
                            "target_is_lag": True,    # lag 로직
                            "y_idx": y_k,
                            "prev_ref": prev_ref
                        })
    return feedback_map

class FeedbackAdapter:
    """
    [BAS 주석] 공정 시뮬레이터의 핵심 엔진 (State State Estimator)
    모델이 예측한 변화량(Diff)을 실제 물리량(Temperature, NOx ppm 등)으로 변환하고,
    이를 다음 타임스텝의 입력값으로 갱신하여 연속적인 시뮬레이션(Rollout)을 가능하게 합니다.
    """
    def __init__(self, mapping):
        self.mapping = mapping  # {src_stage: [ops...]}

    @torch.no_grad()
    def update(self, xs_list, y_hats, x_next_map):
        """
        x_next_map: dict[(stage_idx, stream_idx)] -> Tensor (B, F)  # t+1 실측 입력
        반드시 제공되어야 합니다. 없으면 KeyError.
        """
        # 0) src_stage별로 base(t) = base_1min(t) + diff_pred(t) 계산해 캐시
        # [BAS 설명] y(t) = y(t-1) + delta_y (적분 과정)
        level_cache = {}  # (src_stage, y_idx) -> (B,)
        for src_stage, ops in self.mapping.items():
            y = y_hats[src_stage]  # (B, D_src)
            computed = set()
            for op in ops:
                key = (src_stage, op["y_idx"])
                if key in computed:
                    continue
                pr = op["prev_ref"]  # 오너의 base_1min(t)
                X_prev = xs_list[pr["stage"]][pr["stream"]]       # (B,T_prev,F_prev)
                prev_level_t = X_prev[:, -1, pr["feat_idx"]]      # base_1min(t) == base(t-1)
                level_t = prev_level_t + y[:, op["y_idx"]]        # base(t)
                level_cache[key] = level_t
                computed.add(key)

        # 1) 타깃 스트림별로 묶어서 처리
        buckets = {}
        for src_stage, ops in self.mapping.items():
            for op in ops:
                key = (op["stage"], op["stream"])
                buckets.setdefault(key, []).append((src_stage, op))

        for (tgt_stage, tgt_stream), stream_ops in buckets.items():
            # 필수: X_next(t+1) 존재 확인 및 참조 준비
            try:
                X_next = x_next_map[(tgt_stage, tgt_stream)]  # (B, F)
            except KeyError as e:
                raise KeyError(f"[FeedbackAdapter] x_next_map missing for (stage={tgt_stage}, stream={tgt_stream})") from e

            X_tgt = xs_list[tgt_stage][tgt_stream]            # (B, T, F)
            B, T, F = X_tgt.shape

            # (1) 현재행(t) 현재값(base) 덮어쓰기
            # [BAS 설명] 예측된 현재 상태를 입력 데이터에 반영 (State Update)
            for (src_stage, op) in stream_ops:
                if op["type"] != "diff_to_level":
                    continue
                level_t = level_cache[(src_stage, op["y_idx"])]   # (B,)
                X_tgt[:, -1, op["feat_idx"]] = level_t            # base(t)

            # (2) 미래행(t+1)의 lag(*_1min) 값을 X_next에 먼저 반영
            # [BAS 설명] 현재 상태 t는 다음 스텝 t+1의 '과거값(lag)'이 됨
            for (src_stage, op) in stream_ops:
                if op.get("target_is_lag", False):  # lag 피처면 True로 세팅되어 있음
                    level_t = level_cache[(src_stage, op["y_idx"])]  # base(t)
                    X_next[:, op["feat_idx"]] = level_t              # base_1min(t+1) <- base(t)

            # (3) 시프트 후 마지막 행에 X_next 주입
            # [BAS 설명] Time-Shift: 윈도우를 한 칸 미래로 이동
            if T > 1:
                X_tgt[:, :-1, :] = X_tgt[:, 1:, :]
            X_tgt[:, -1, :] = X_next  # (B, F)


def make_kpm(lengths, max_len):
    device = lengths.device
    idx = torch.arange(max_len, device=device).unsqueeze(0)
    valid = idx < lengths.unsqueeze(1)     # (B,T) True=valid
    return ~valid                          # True=PAD

def run_gru_packed(gru: nn.GRU, x_padded, lengths):
    # [BAS 설명] 가변 길이 시계열 데이터를 효율적으로 처리하기 위한 Packing
    packed = pack_padded_sequence(x_padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
    packed_out, _ = gru(packed)
    H, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B,T,hidden*num_dir)
    return H

def run_gru_fast(gru: nn.GRU, x_padded, lengths):
    T = x_padded.size(1)
    if (lengths == T).all():
        H, _ = gru(x_padded)     # pack/unpack 생략 (고정 길이)
        return H
    packed = pack_padded_sequence(x_padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
    packed_out, _ = gru(packed)
    H, _ = pad_packed_sequence(packed_out, batch_first=True)
    return H

def left_pad_time(X: torch.Tensor, kpm: torch.Tensor, T_max: int):
    """
    X: (B, T, d), kpm: (B, T)  [True=PAD]
    반환: X_pad: (B, T_max, d), kpm_pad: (B, T_max)
    짧은 시퀀스를 앞쪽(과거)으로 패딩해서 끝(t) 시점을 정렬합니다.
    [BAS 설명] 화학 공정 데이터는 '현재 시점(t)' 기준 정렬이 중요하므로,
    길이가 다른 시퀀스는 과거 부분을 패딩 처리하여 끝을 맞춥니다.
    """
    B, T, d = X.shape
    if T == T_max:
        return X, kpm
    pad_len = T_max - T
    pad_X   = torch.zeros(B, pad_len, d, device=X.device, dtype=X.dtype)
    pad_kpm = torch.ones(B, pad_len, device=kpm.device, dtype=kpm.dtype)  # True=PAD
    X_pad   = torch.cat([pad_X, X], dim=1)
    kpm_pad = torch.cat([pad_kpm, kpm], dim=1)
    return X_pad, kpm_pad

# =======================
# Blocks (수정됨)
# =======================
class CrossAttentionBlock(nn.Module):
    """
    [BAS 주석] 공정 단계 간 상호작용 모델링
    물리적으로 떨어진 공정 단계(예: 1단 온도 -> TMS) 간의 상관관계를
    Attention 메커니즘으로 포착합니다.
    Upstream의 상태 변화가 Downstream에 도달하는 '지연 시간(Dead time)'을
    학습하기 위해 사용됩니다.
    """
    def __init__(self, d_model, nhead=8, dropout=0.2): # [수정] 기본값 0.1 -> 0.2 상향 권장
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln1  = nn.LayerNorm(d_model)
        # [수정] FFN 내부 Dropout 유지
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, 4*d_model), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Linear(4*d_model, d_model)
        )
        self.ln2  = nn.LayerNorm(d_model)
        self.dropout_val = dropout # 디버깅용 저장

    def forward(self, Q, K, V, key_padding_mask=None, attn_mask=None):
        ctx, _ = self.attn(Q, K, V, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # Residual Connection에도 Dropout을 거는 것이 일반적이나, 여기서는 기존 구조 유지
        X = self.ln1(Q + ctx) 
        X = self.ln2(X + self.ffn(X))
        return X

class GatedFusionStepwise(nn.Module):
    """ 
    시점별 게이팅: 여러 스트림 (동일한 타임라인으로 pad되어 있음)을 가볍게 합침
    [BAS 설명] Sensor Fusion Layer.
    같은 공정 단계 내의 이종 센서(예: 온도 센서 + 압력 센서) 데이터를
    중요도(Gate)에 따라 가중 결합합니다.
    """
    def __init__(self, d_model):
        super().__init__()
        self.g1 = nn.Sequential(nn.Linear(2*d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
    def forward(self, streams: List[torch.Tensor], mask: torch.Tensor = None):
        """
        streams: list of (B,T,d)  [S개 스트림]
        mask: (B,T) optional (유효시점만 남길 때)
        """
        assert len(streams) >= 1
        if len(streams) == 1:
            fused = streams[0]
        else:
            A, B = streams[0], streams[1]
            G = torch.sigmoid(self.g1(torch.cat([A, B], dim=-1)))  # (B,T,1)
            fused = G * A + (1 - G) * B
            # 3개 이상이면 순차적으로 fuse
            for s in streams[2:]:
                G = torch.sigmoid(self.g1(torch.cat([fused, s], dim=-1)))
                fused = G * fused + (1 - G) * s
        if mask is not None:
            fused = fused * mask.unsqueeze(-1)
        return fused

# =======================
# Multi-Stage Model (수정됨)
# =======================
class MultiStageProcessModel(nn.Module):
    """
    [BAS 핵심 주석] 계층적 화학 공정 모델 (Hierarchical Process Model)
    이 모델은 실제 공장의 물리적 배치(Layout)를 그대로 신경망 구조로 옮겼습니다.

    Stage 0 (Reactor 1) -> Stage 1 (Reactor 2) -> Stage 2 (Analyzer) -> Stage 3 (TMS/Stack)
    
    각 단계는:
    1. GRU: 해당 단계의 시계열적 동특성(Dynamics) 학습
    2. Gated Fusion: 다중 센서 데이터 통합
    3. Cross Attention: 이전 단계(Upstream) 정보 참조 (물리적 흐름 반영)
    4. Head: 해당 단계의 주요 인자(온도, 농도 등) 예측
    """
    def __init__(self, stage_specs: List[Dict[str, Any]],
                 hidden=128, d_model=128, num_layers=1, bidirectional=False, nhead=8, 
                 dropout=0.2): # [수정] dropout 인자 추가 및 기본값 상향 (0.2~0.3 권장)
        super().__init__()
        self.specs = stage_specs
        self.N = len(stage_specs)
        self.num_dir = 2 if bidirectional else 1
        enc_dim = hidden * self.num_dir

        # 1. 스트림별 GRU (Thermal Inertia / Chemical Lag 학습)
        self.stream_grus = nn.ModuleList()
        for spec in stage_specs:
            stage_grus = nn.ModuleList([
                # [수정] GRU에 dropout 인자 전달
                nn.GRU(in_dim, hidden, num_layers=num_layers, batch_first=True, 
                       bidirectional=bidirectional, dropout=(dropout if num_layers > 1 else 0))
                for in_dim in spec['streams']
            ])
            self.stream_grus.append(stage_grus)

        # 2. 스트림 Projection
        self.stream_projs = nn.ModuleList()
        for spec in stage_specs:
            projs = nn.ModuleList([
                nn.Linear(enc_dim, d_model) if enc_dim != d_model else nn.Identity()
                for _ in spec['streams']
            ])
            self.stream_projs.append(projs)

        # 3. Gated Fusion (Sensor Fusion)
        self.stage_fusions = nn.ModuleList([
            GatedFusionStepwise(d_model) for _ in stage_specs
        ])

        # 4. Cross-Attn (Flow Propagation: Upstream -> Downstream)
        self.xattn_blocks = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead=nhead, dropout=dropout) if k > 0 else nn.Identity()
            for k in range(self.N)
        ])

        # 5. 단계별 Head (여기가 핵심!)
        self.heads = nn.ModuleList()
        for spec in stage_specs:
            self.heads.append(nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout), 
                nn.Linear(d_model, spec['out_dim'])
            ))
            
    def forward(self, xs_list, lens_list, max_stage: int = None):
        assert len(xs_list) == self.N and len(lens_list) == self.N
        N_run = self.N if (max_stage is None) else (max_stage + 1)

        stage_H = []    
        stage_kpm = []    
        for k in range(N_run):
            spec = self.specs[k]
            stream_Hs, stream_kpms = [], []
            for s_idx, _ in enumerate(spec['streams']):
                H_enc = run_gru_fast(self.stream_grus[k][s_idx], xs_list[k][s_idx], lens_list[k][s_idx])
                H_proj = self.stream_projs[k][s_idx](H_enc)                                                        
                stream_Hs.append(H_proj)
                stream_kpms.append(make_kpm(lens_list[k][s_idx], H_proj.size(1)))                                    

            T_max = max(h.size(1) for h in stream_Hs)
            
            padded_Hs, padded_kpms = [], []
            for H_s, kpm_s in zip(stream_Hs, stream_kpms):
                H_pad, kpm_pad = left_pad_time(H_s, kpm_s, T_max)
                padded_Hs.append(H_pad)       
                padded_kpms.append(kpm_pad) 
            
            valid_any = None
            for kpm_pad in padded_kpms:
                valid = (~kpm_pad).float()                  
                valid_any = valid if valid_any is None else torch.clamp(valid_any + valid, max=1.0)
            valid_any = valid_any.bool()                    
            
            stage_fused = self.stage_fusions[k](padded_Hs, mask=valid_any) 
            stage_H.append(stage_fused)
            stage_kpm.append(~valid_any)                  

        y_hats, stage_reprs = [], []

        for k in range(N_run):
            spec = self.specs[k]
            Q = stage_H[k]
            if k == 0:
                fused = Q
            else:
                # [BAS 설명] Cross Attention: 이전 단계(0~k-1)의 정보를 Key, Value로 참조
                # 즉, TMS(k) 예측 시 반응기(0,1)의 과거 상태를 조회함
                K = torch.cat(stage_H[:k], dim=1)                 
                kpm = torch.cat(stage_kpm[:k], dim=1)             
                fused = self.xattn_blocks[k](Q, K, K, key_padding_mask=kpm)

            mask_valid = (~stage_kpm[k]).float().unsqueeze(-1)    
            denom = mask_valid.sum(1).clamp_min(1e-9)             
            pooled = (fused * mask_valid).sum(1) / denom          
            yk = self.heads[k](pooled)                             
            y_hats.append(yk)
            stage_reprs.append(fused)

        return y_hats, stage_reprs

@torch.inference_mode()
def predict_until(model, xs_list, lens_list, last_stage: int):
    model.eval()
    y_hats, _ = model(xs_list, lens_list, max_stage=last_stage)
    return y_hats  # 0..last_stage까지만 들어 있음

def predict_until_with_dropout(model, xs_list, lens_list, last_stage: int):
    # [BAS 설명] MC Dropout: 불확실성(Uncertainty) 추정을 위해 Inference 시에도 Dropout 활성화
    model.eval()
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.MultiheadAttention)):
            m.train()
    with torch.no_grad():
        y_hats, _ = model(xs_list, lens_list, max_stage=last_stage)
    return y_hats 

# =======================
# Toy Dataset (다중 스트림/다변량 타깃)
# =======================
class MultiStreamDataset(Dataset):
    """
    [BAS 주석] 화학 공정 데이터셋
    - Rolling Window 방식으로 시계열 데이터를 슬라이싱
    - 공정 조건(LNG 투입, 질산 투입 등)을 별도 조건(Condition)으로 관리
    """
    def __init__(self, data, stage_specs, 인풋변수s, 타깃변수s, min_seq_len = 10, max_seq_len = 100, rollout_horizon: int = 10):
        self.stage_specs = stage_specs
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.rollout_horizon = rollout_horizon

        # [데이터 로딩부 - Pandas DataFrame에서 Numpy로 변환]
        self.X1_1 = data.loc[:, 인풋변수s['1단 온도_1'][0]].to_numpy(dtype=np.float32) 
        self.X1_2 = data.loc[:, 인풋변수s['1단 온도_1'][1]].to_numpy(dtype=np.float32)  
        self.X2_1 = data.loc[:, 인풋변수s['3단 온도_1'][0]].to_numpy(dtype=np.float32)  
        self.X3_1 = data.loc[:, 인풋변수s['Analyzer'][0]].to_numpy(dtype=np.float32)  
        self.X3_2 = data.loc[:, 인풋변수s['Analyzer'][1]].to_numpy(dtype=np.float32)  
        self.X4_1 = data.loc[:, 인풋변수s['TMS'][0]].to_numpy(dtype=np.float32)  
        self.X4_2 = data.loc[:, 인풋변수s['TMS'][1]].to_numpy(dtype=np.float32)  

        self.y1_1 = data.loc[:, 타깃변수s['1단 온도_1'][0]].to_numpy(dtype=np.float32)  
        self.y1_2 = data.loc[:, 타깃변수s['1단 온도_1'][1]].to_numpy(dtype=np.float32)  
        
        
        self.y2_1 = data.loc[:, 타깃변수s['3단 온도_1'][0]].to_numpy(dtype=np.float32)  
        
        self.y3_1 = data.loc[:, 타깃변수s['Analyzer'][0]].to_numpy(dtype=np.float32)  
        self.y3_2 = data.loc[:, 타깃변수s['Analyzer'][1]].to_numpy(dtype=np.float32)  
        self.y3_3 = data.loc[:, 타깃변수s['Analyzer'][2]].to_numpy(dtype=np.float32)  

        self.y4_1 = data.loc[:, 타깃변수s['TMS'][0]].to_numpy(dtype=np.float32)  
        self.y4_2 = data.loc[:, 타깃변수s['TMS'][1]].to_numpy(dtype=np.float32)  
        self.y4_3 = data.loc[:, 타깃변수s['TMS'][2]].to_numpy(dtype=np.float32)  
        self.y4_4 = data.loc[:, 타깃변수s['TMS'][3]].to_numpy(dtype=np.float32)  
        self.y4_5 = data.loc[:, 타깃변수s['TMS'][4]].to_numpy(dtype=np.float32)  
        self.y4_6 = data.loc[:, 타깃변수s['TMS'][5]].to_numpy(dtype=np.float32)  
        self.y4_7 = data.loc[:, 타깃변수s['TMS'][6]].to_numpy(dtype=np.float32)  

        # [이벤트 변수] 공정의 급격한 변화를 유발하는 트리거(Trigger) 변수들
        self.cond_lng = data['LNG변화_경과시간_분'].to_numpy(dtype=np.float32)
        self.cond_514 = data['514질산_시작경과시간_분'].to_numpy(dtype=np.float32)
        # 518 질산은 1,2,3 중 하나라도 양수면 되므로 미리 max로 합쳐두면 효율적입니다.
        self.cond_518 = data[['518질산1_시작경과시간_분', '518질산2_시작경과시간_분', '518질산3_시작경과시간_분']].max(axis=1).to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.X1_1) - self.max_seq_len - self.rollout_horizon

    def __getitem__(self, idx):
        H = self.rollout_horizon
        t_now = idx + self.max_seq_len
        X = {
            'x1_1': torch.from_numpy(self.X1_1[t_now-self.min_seq_len:t_now]),
            'x1_2': torch.from_numpy(self.X1_2[idx:t_now]),
            'x2_1': torch.from_numpy(self.X2_1[t_now-self.min_seq_len:t_now]),
            'x3_1': torch.from_numpy(self.X3_1[t_now-self.min_seq_len:t_now]),
            'x3_2': torch.from_numpy(self.X3_2[idx:t_now]),
            'x4_1': torch.from_numpy(self.X4_1[t_now-self.min_seq_len:t_now]),
            'x4_2': torch.from_numpy(self.X4_2[idx:t_now])
        }
        y = {
            'y1': {
                '1단온도_1': torch.tensor(self.y1_1[t_now]),
                '1단온도_2': torch.tensor(self.y1_2[t_now]),
            },
            'y2': {
                '3단온도_1': torch.tensor(self.y2_1[t_now]),
            },
            'y3': {
                'CO': torch.tensor(self.y3_1[t_now]),
                'Nox': torch.tensor(self.y3_2[t_now]),
                'O2': torch.tensor(self.y3_3[t_now]),
            },
            'y4': {
                'CO': torch.tensor(self.y4_1[t_now]),
                'Nox': torch.tensor(self.y4_2[t_now]),
                'O2': torch.tensor(self.y4_3[t_now]),
                'NO': torch.tensor(self.y4_4[t_now]),
                'NO2': torch.tensor(self.y4_5[t_now]),
                'Flow': torch.tensor(self.y4_6[t_now]),
                'Temp': torch.tensor(self.y4_7[t_now]),
            }
        }

        y_future = {
            'y1': torch.stack([
                _slice_future(self.y1_1, t_now, H),
                _slice_future(self.y1_2, t_now, H),
                
            ], dim=1), # (H,3)
            'y2': torch.stack([
                _slice_future(self.y2_1, t_now, H),
            ], dim=1), # (H,2)
            'y3': torch.stack([
                _slice_future(self.y3_1, t_now, H),
                _slice_future(self.y3_2, t_now, H),
                _slice_future(self.y3_3, t_now, H),
            ], dim=1), # (H,3)
            'y4': torch.stack([
                _slice_future(self.y4_1, t_now, H),
                _slice_future(self.y4_2, t_now, H),
                _slice_future(self.y4_3, t_now, H),
                _slice_future(self.y4_4, t_now, H),
                _slice_future(self.y4_5, t_now, H),
                _slice_future(self.y4_6, t_now, H),
                _slice_future(self.y4_7, t_now, H),
            ], dim=1), # (H,7)
        }

        X_future = {
            # stage 0: '1단 온도_1' (streams: 2)
            (0,0): _slice_future(self.X1_1, t_now, H), # (H, F)
            (0,1): _slice_future(self.X1_2, t_now, H),
            # stage 1: '3단 온도_1' (streams: 1)
            (1,0): _slice_future(self.X2_1, t_now, H),
            # stage 2: 'Analyzer' (streams: 2)
            (2,0): _slice_future(self.X3_1, t_now, H),
            (2,1): _slice_future(self.X3_2, t_now, H),
            # stage 3: 'TMS' (streams: 2)
            (3,0): _slice_future(self.X4_1, t_now, H),
            (3,1): _slice_future(self.X4_2, t_now, H),
        }

        # [추가] 현재 시점(t_now) 및 미래 시점(future)의 가중치 조건 추출
        w_conditions = {
            'lng': torch.tensor(self.cond_lng[t_now]),
            '514': torch.tensor(self.cond_514[t_now]),
            '518': torch.tensor(self.cond_518[t_now])
        }
        w_future_conditions = {
            'lng': _slice_future(self.cond_lng, t_now, H), # (H,)
            '514': _slice_future(self.cond_514, t_now, H), # (H,)
            '518': _slice_future(self.cond_518, t_now, H) # (H,)
        }
        return X, y, y_future, X_future, w_conditions, w_future_conditions


def _ensure_2d(x: torch.Tensor):
    # (T,) -> (T,1) 로 변환, 이미 (T,F)이면 그대로
    return x.unsqueeze(-1) if x.ndim == 1 else x

def collate_from_custom_dataset(batch):
    # [BAS 설명] 배치 단위로 데이터를 묶는 함수 (Pytorch DataLoader 용)
    B = len(batch)

    # ---- 1) X를 stage별/stream별로 모으기 ----
    # Stage1
    s1_x1 = [ _ensure_2d(batch[b][0]['x1_1']) for b in range(B) ] # 10
    s1_x2 = [ _ensure_2d(batch[b][0]['x1_2']) for b in range(B) ] # 100
    # Stage2
    s2_x1 = [ _ensure_2d(batch[b][0]['x2_1']) for b in range(B) ] # 10
    # Stage3
    s3_x1 = [ _ensure_2d(batch[b][0]['x3_1']) for b in range(B) ] # 10
    s3_x2 = [ _ensure_2d(batch[b][0]['x3_2']) for b in range(B) ] # 100
    # Stage4
    s4_x1 = [ _ensure_2d(batch[b][0]['x4_1']) for b in range(B) ] # 10
    s4_x2 = [ _ensure_2d(batch[b][0]['x4_2']) for b in range(B) ] # 100

    # 배치 스택 
    def stack_batch(seqs): # seqs: list of (T,F)
        T = seqs[0].size(0)
        return torch.stack(seqs, dim=0) # (B,T,F)

    xs_list = [
        [ stack_batch(s1_x1), stack_batch(s1_x2) ], # Stage1: 2 streams
        [ stack_batch(s2_x1) ], # Stage2: 1 stream
        [ stack_batch(s3_x1), stack_batch(s3_x2) ], # Stage3: 1 stream
        [ stack_batch(s4_x1), stack_batch(s4_x2) ], # Stage4: 2 streams
    ]

    # ---- 2) 길이 텐서 (모두 고정 길이) ----
    lens_list = []
    for stage_streams in xs_list:
        stage_lens = []
        for Xs in stage_streams:
            T = Xs.size(1)
            stage_lens.append(torch.full((B,), T, dtype=torch.long))
        lens_list.append(stage_lens)

    # ---- 3) y를 stage별 out_dim으로 묶기 ----
    y1 = torch.stack([ torch.stack([
                batch[b][1]['y1']['1단온도_1'],
                batch[b][1]['y1']['1단온도_2'],
            ], dim=0) for b in range(B)
        ], dim=0).float() # (B,3)
    y2 = torch.stack([ torch.stack([
                batch[b][1]['y2']['3단온도_1'],
            ], dim=0) for b in range(B)
        ], dim=0).float() # (B,2)
    y3 = torch.stack([ torch.stack([
                batch[b][1]['y3']['CO'],
                batch[b][1]['y3']['Nox'],
                batch[b][1]['y3']['O2'],
            ], dim=0) for b in range(B)
        ], dim=0).float() # (B,3)

    y4 = torch.stack([ torch.stack([
                batch[b][1]['y4']['CO'],
                batch[b][1]['y4']['Nox'],
                batch[b][1]['y4']['O2'],
                batch[b][1]['y4']['NO'],
                batch[b][1]['y4']['NO2'],
                batch[b][1]['y4']['Flow'],
                batch[b][1]['y4']['Temp'],
            ], dim=0) for b in range(B)
        ], dim=0).float() # (B,7)

    ys_list = [ y1, y2, y3, y4 ] # 각 (B, out_dim_k)

    # ---- 추가: ys_future_list (B,H,D) ----
    ys_future_list = []
    # stage1
    ys_future_list.append(torch.stack([batch[b][2]['y1'] for b in range(B)], dim=0).float()) # (B,H,3)
    # stage2
    ys_future_list.append(torch.stack([batch[b][2]['y2'] for b in range(B)], dim=0).float()) # (B,H,2)
    # stage3
    ys_future_list.append(torch.stack([batch[b][2]['y3'] for b in range(B)], dim=0).float()) # (B,H,3)
    # stage4
    ys_future_list.append(torch.stack([batch[b][2]['y4'] for b in range(B)], dim=0).float()) # (B,H,7)

    # ---- X_future를 (B,H,F)로 쌓기 ----
    def stack_X_future(stage, stream):
        return torch.stack([batch[b][3][(stage, stream)] for b in range(B)], dim=0).float() # (B,H,F)

    xs_future_list = [
        [ stack_X_future(0,0), stack_X_future(0,1) ], # Stage1
        [ stack_X_future(1,0) ], # Stage2
        [ stack_X_future(2,0), stack_X_future(2,1) ], # Stage3
        [ stack_X_future(3,0), stack_X_future(3,1) ], # Stage4
    ]

    # [추가] 가중치 조건 스택
    w_cond_batch = {
        'lng': torch.stack([batch[b][4]['lng'] for b in range(B)]),
        '514': torch.stack([batch[b][4]['514'] for b in range(B)]),
        '518': torch.stack([batch[b][4]['518'] for b in range(B)])
    }

    w_future_batch = {
        'lng': torch.stack([batch[b][5]['lng'] for b in range(B)]),
        '514': torch.stack([batch[b][5]['514'] for b in range(B)]),
        '518': torch.stack([batch[b][5]['518'] for b in range(B)])
    }

    return xs_list, lens_list, ys_list, ys_future_list, xs_future_list, w_cond_batch, w_future_batch

def infer_stage_specs_from_batch(xs_list, ys_list):
    specs = []
    for k, stage_streams in enumerate(xs_list):
        in_dims = [ Xs.size(2) for Xs in stage_streams ] # F per stream
        out_dim = ys_list[k].size(1)
        specs.append({"streams": in_dims, "out_dim": out_dim})
    return specs

# =======================
# Trainer (다변량 + 단계 가중치)
# =======================
class Trainer:
    """
    [BAS 핵심 주석] 훈련 및 제어 로직 (Controller Logic)
    이 클래스는 단순 학습뿐만 아니라,
    1) Lower Bound Model을 이용한 Anomaly Detection (이상치 마스킹)
    2) Event-based Dynamic Weighting (LNG 변경, 질산 투입 시 가중치 부여)
    3) Stage-wise Curriculum Learning (하류 공정부터 상류 공정 순차 학습)
    을 수행합니다.
    """
    def __init__(self, model, stage_weights, lr=3e-4, wd=1e-4, grad_clip=1.0, use_amp=True, device=None,
                 rollout_horizon: int = 10, rollout_weight: float = 0.5,
                 feedback_adapter: FeedbackAdapter = None,
                 rollout_discount: float = 1.0,
                 anomaly_mask_ratio: float = 0.2):  # [추가] 하위 20%를 이상치로 간주하여 마스킹
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # 메인 모델 설정
        self.model = model.to(device)
        
        # [핵심] Lower Bound Model (Target Network와 유사)
        # 안정적인 학습 기준점(Baseline)을 제공하기 위해 메인 모델의 '과거 베스트 버전'을 유지합니다.
        # Reinforcement Learning의 Target Network 아이디어를 차용했습니다.
        self.lb_model = copy.deepcopy(model).to(device)
        self.lb_model.eval()
        # LB 모델은 학습되지 않도록 Gradient 계산 비활성화
        for p in self.lb_model.parameters():
            p.requires_grad = False
            
        self.mask_ratio = anomaly_mask_ratio
        
        # 최적화기 설정
        self.lr = lr
        self.wd = wd
        self.opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode="min", factor=0.5, patience=30, min_lr=1e-8
        )
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        # 가중치 관련 설정
        self.rollout_horizon = rollout_horizon
        self.rollout_weight = rollout_weight
        self.feedback_adapter = feedback_adapter
        self.rollout_discount = rollout_discount
        self.stage_weight_scale = [1.0]*len(stage_weights)
        self.phase = '1단온도'
        
        # Loss 함수 (Reduction='none' 필수: 샘플별 마스킹 적용을 위해)
        self.crit = nn.MSELoss(reduction="none")
        self.stage_weights_base = stage_weights
        self.active_stages = None
        self.stage_weights = stage_weights

    def set_active_stages(self, active_set):
        self.active_stages = set(active_set)

    def set_weight_scale(self, scales):
        self.stage_weight_scale = scales
        
    def _update_lb_model(self):
        """
        [핵심] Validation Best 갱신 시 호출.
        메인 모델의 가중치를 LB 모델로 복사 (Target Network Update)
        """
        self.lb_model.load_state_dict(self.model.state_dict())
        self.lb_model.eval()
        # print("[Trainer] Lower Bound Model updated with new best weights.")

    def _to_device(self, batch):
        # (기존 코드와 동일)
        xs_list, lens_list, ys_list, ys_future_list, xs_future_list, w_cond, w_future = batch
        xs_list = [[x.to(self.device) for x in stage_streams] for stage_streams in xs_list]
        lens_list = [[l.to(self.device) for l in stage_lens] for stage_lens in lens_list]
        ys_list = [y.to(self.device) for y in ys_list]
        ys_future_list = [y.to(self.device) for y in ys_future_list]
        xs_future_list = [[xf.to(self.device) for xf in stage_streams] for stage_streams in xs_future_list]
        w_cond = {k: v.to(self.device) for k, v in w_cond.items()}
        w_future = {k: v.to(self.device) for k, v in w_future.items()}
        return xs_list, lens_list, ys_list, ys_future_list, xs_future_list, w_cond, w_future

    def _forward_until(self, xs_list, lens_list, last_stage: int):
        # model.forward(..., max_stage=last_stage)로 grads 유지
        y_hats, _ = self.model(xs_list, lens_list, max_stage=last_stage)
        return y_hats[last_stage] # (B, D_last)

    def _apply_stage_update_no_shift(self, src_stage: int, xs_list, y_hat_stage, x_next_map):
        """
        src_stage의 예측 y_hat_stage(=diff)를 이용해:
        (1) target_is_lag=False → xs_list의 현재행(base) 덮어쓰기
        (2) target_is_lag=True → x_next_map의 해당 lag 피처 덮어쓰기
        """
        ops = self.feedback_adapter.mapping.get(src_stage, [])
        for op in ops:
            # base(t) = base_1min(t) + diff_pred(t)
            pr = op["prev_ref"] # 오너의 base_1min (시점 t)
            X_prev = xs_list[pr["stage"]][pr["stream"]]
            prev_level_t = X_prev[:, -1, pr["feat_idx"]] # (B,)
            level_t = prev_level_t + y_hat_stage[:, op["y_idx"]] # (B,)
            if op.get("target_is_lag", False):
                # 미래행 lag: X_next에 미리 반영
                X_next = x_next_map[(op["stage"], op["stream"])] # (B,F)
                X_next[:, op["feat_idx"]] = level_t
            else:
                # 현재행 base: xs_list 현재행(-1)에 덮어쓰기
                X_tgt = xs_list[op["stage"]][op["stream"]] # (B,T,F)
                X_tgt[:, -1, op["feat_idx"]] = level_t
                
    def _rollout_losses_stagewise(self, xs_list, lens_list, ys_future_list, xs_future_list, w_future_batch, max_stage_idx=None):
        """
        [BAS 주석] 가상 공정 운전 (Virtual Process Run)
        한 step(h) 안에서 stage 0→1→2→3 순서로:
        - forward_until(stage k) → 예측 수행 및 손실 계산
        - 현재값 업데이트 및 미래값(Lag) 준비
        - 공정 전체를 1스텝 전진(Time Shift)
        
        이 과정은 실제 공장의 운전을 가상으로 모사하며, 미래의 에러가 
        현재의 학습에 영향을 주는 Backpropagation through time(BPTT)과 유사합니다.
        """
        assert self.feedback_adapter is not None, "FeedbackAdapter를 설정하세요."
        H = self.rollout_horizon
        total = torch.tensor(0.0, device=self.device)
        gamma = 1.0
        num_stages = len(xs_list)
        last_stage = (max_stage_idx if max_stage_idx is not None else num_stages - 1)
        for h in range(H):
            # (a) 이번 step용 X_next 맵 만들기: (stage,stream) -> (B,F)
            x_next_map = {}
            for k in range(num_stages):
                for j in range(len(xs_future_list[k])):
                    # xs_future_list[k][j]: (B,H,F) → 이번 step h의 (B,F)
                    x_next_map[(k,j)] = xs_future_list[k][j][:, h, :].clone()

            # [추가] 현재 스텝(h)의 미래 가중치 조건 추출 (B,)
            current_future_cond = {
                'lng': w_future_batch['lng'][:, h],
                '514': w_future_batch['514'][:, h],
                '518': w_future_batch['518'][:, h]
            }
            # (b) stage-by-stage 순차 예측 & 반영 (시프트 X)
            for k in range(last_stage + 1):
                if (self.active_stages is not None) and (k not in self.active_stages):
                    continue
                # forward until k (grads 유지)
                yk = self._forward_until(xs_list, lens_list, last_stage=k) # (B, D_k)
                # rollout 타깃은 미래 시점: t0+1+h
                yt_h = ys_future_list[k][:, h, :] # (B, D_k)
                # 가중 손실
                base_w = self.stage_weights_base[k]
                alpha = self.stage_weight_scale[k]
                scaled_w = (torch.tensor(base_w, dtype=torch.float32, device=self.device)
                            if isinstance(base_w, (list, tuple)) else base_w)
                if isinstance(scaled_w, torch.Tensor):
                    scaled_w = scaled_w * alpha
                else:
                    scaled_w = float(scaled_w) * alpha

                # 2. [추가] 미래 시점(h)의 샘플별 동적 가중치 계산
                sample_w = self._get_dynamic_weight(k, current_future_cond, high_weight_value=10.0)

                # 3. 최종 가중치
                final_w = scaled_w * sample_w.unsqueeze(-1)
                total = total + gamma * self._weighted_loss(yk, yt_h, final_w, self.device)
                
                # (1) 현재행(base) 덮어쓰기 + (2) X_next lag 덮어쓰기 (시프트는 하지 않음)
                self._apply_stage_update_no_shift(k, xs_list, yk, x_next_map)
                
            # (c) 모든 스트림 한 번만 시프트하고 마지막 행을 X_next로
            for k in range(num_stages):
                for j in range(len(xs_list[k])):
                    X_tgt = xs_list[k][j] # (B,T,F)
                    if X_tgt.size(1) > 1:
                        X_tgt[:, :-1, :] = X_tgt[:, 1:, :]
                    X_tgt[:, -1, :] = x_next_map[(k,j)]
            # 감가 계수
            gamma *= self.rollout_discount
        return total / H

    def _compute_anomaly_mask(self, yh, y_lb, yt):
        """
        [핵심] Anomaly Mask 생성
        S(X) = |yt - yh| (현재 모델 잔차) - |yt - y_lb| (Best 모델 잔차)
        S 값이 작다는 것은 '현재 모델도 못 맞추고 Best 모델도 못 맞춤' -> 이상치일 확률 높음.
        하위 r%를 마스킹.
        [BAS 설명] 화학 공정 데이터에는 센서 고장이나 물리적으로 설명 불가능한 
        이상치(Outlier)가 빈번합니다. 이 로직은 모델이 도저히 학습할 수 없는
        '나쁜 데이터'를 학습에서 배제(Masking)하여 모델의 강건성을 높입니다.
        """
        # 잔차 계산 (절대값 오차)
        res_curr = torch.abs(yt - yh)       # (B, D)
        res_lb = torch.abs(yt - y_lb)       # (B, D)
        
        # S 점수: (Unlearned - Anomaly)
        # S가 크다 = 현재 모델은 많이 틀리는데, Best 모델은 잘 맞췄음 -> 학습 기회 (Unlearned)
        # S가 작다(~0) = 둘 다 많이 틀림 -> 이상치 (Anomaly)
        s_score = res_curr - res_lb         # (B, D)
        
        # 채널(변수) 독립적으로 마스킹 (논문 구현 방식)
        # 배치(dim=0) 차원에서 하위 r% 임계값 계산
        B = s_score.shape[0]
        k = int(B * self.mask_ratio)
        if k < 1: 
            return torch.ones_like(s_score) # 배치가 너무 작으면 마스킹 안함

        # kthvalue는 오름차순 정렬 후 k번째 값 (즉, 하위 r% 경계값)
        threshold, _ = torch.kthvalue(s_score, k, dim=0, keepdim=True) # (1, D)
        
        # S 점수가 임계값보다 큰 애들만 살림 (1), 작은 애들은 죽임 (0)
        mask = (s_score >= threshold).float()
        return mask
    def _get_dynamic_weight(self, stage_idx, cond_dict, high_weight_value=10.0):
        """
        stage_idx와 조건(cond_dict)에 따라 샘플별 가중치(B,)를 계산
        cond_dict: {'lng': (B,), '514': (B,), ...}
        [BAS 설명] 도메인 지식 반영:
        - LNG 투입 변화 시점은 1단/3단 온도 제어에 매우 중요하므로 가중치 10배
        - 질산 투입(514/518) 시점은 NOx 발생의 원인이므로 Analyzer/TMS 학습에 가중치 부여
        """
        device = self.device
        B = cond_dict['lng'].size(0)
        # 기본 가중치 1.0으로 시작
        weights = torch.ones(B, device=device, dtype=torch.float32)
        # 1단 온도(0), 3단 온도(1) 인 경우
        if stage_idx in [0, 1]:
            # LNG 변화 경과시간이 양수면 높은 가중치
            is_event = cond_dict['lng'] > 0
            weights = torch.where(is_event, torch.tensor(high_weight_value, device=device), weights)
        # Analyzer(2), TMS(3) 인 경우
        elif stage_idx in [2, 3]:
            # 514 질산 투입 시 +3
            mask_514 = cond_dict['514'] > 0
            weights = weights + torch.where(mask_514, torch.tensor(3.0, device=device), torch.tensor(0.0, device=device))
            # 518 질산 투입 시 +2
            mask_518 = cond_dict['518'] > 0
            weights = weights + torch.where(mask_518, torch.tensor(2.0, device=device), torch.tensor(0.0, device=device))
        return weights

    @staticmethod
    def _as_tensor_w(w, device, out_dim=None, batch_size=None):
        """
        다양한 형태의 weight를 텐서로 정규화:
        - float/int -> scalar tensor
        - list/tuple -> (D,)
        - torch.Tensor -> 그대로
        """
        if isinstance(w, (int, float)):
            wt = torch.tensor(w, dtype=torch.float32, device=device)
        elif isinstance(w, (list, tuple)):
            wt = torch.tensor(w, dtype=torch.float32, device=device)
        elif isinstance(w, torch.Tensor):
            wt = w.to(device=device, dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported weight type: {type(w)}")

        # shape 검증(가능한 경우에 한해서)
        if wt.ndim == 0:
            pass # scalar
        elif wt.ndim == 1 and out_dim is not None:
            assert wt.numel() == out_dim, f"weight length {wt.numel()} != out_dim {out_dim}"
        elif wt.ndim == 2 and (batch_size is not None and out_dim is not None):
            assert wt.shape[0] == batch_size and wt.shape[1] == out_dim, \
                f"weight shape {wt.shape} must be (B,{out_dim})"
        elif wt.ndim == 1 and batch_size is not None and out_dim is None:
            assert wt.numel() == batch_size, f"weight length {wt.numel()} != batch_size {batch_size}"
        return wt

    def _weighted_loss(self, yh, yt, w, device):
        """
        yh, yt: (B, D) 또는 (B, 1)
        w: scalar | (D,) | (B,) | (B,D)
        """
        B, D = yh.shape[0], yh.shape[1]
        l = self.crit(yh, yt) # (B,D)

        wt = self._as_tensor_w(w, device, out_dim=D, batch_size=B)

        if wt.ndim == 0:
            # scalar
            loss = wt * l.mean()
        elif wt.ndim == 1:
            if wt.numel() == D:
                # per-dim
                loss = (l.mean(dim=0) * wt).mean() 
            elif wt.numel() == B:
                # per-sample
                loss = (l.mean(dim=1) * wt).mean()
            else:
                raise ValueError("1D weight must be length D or B")
        elif wt.ndim == 2:
            # per-sample-per-dim
            loss = (l * wt).mean()
        else:
            raise ValueError("weight tensor must be 0D/1D/2D")
        return loss

    def _one_step_losses(self, y_hats, y_hats_lb, ys_list, w_cond_batch):
        """
        [수정] y_hats_lb (LB 모델 예측)를 추가로 받아서 마스킹 적용
        """
        losses = []
        # y_hats와 y_hats_lb는 stage별 리스트
        for k, (yh, y_lb, yt) in enumerate(zip(y_hats, y_hats_lb, ys_list)):
            if (self.active_stages is not None) and (k not in self.active_stages):
                continue
            
            # 1. 기본 가중치
            base_w = self.stage_weights_base[k]
            alpha = self.stage_weight_scale[k]
            scaled_w = (torch.tensor(base_w, dtype=torch.float32, device=self.device)
                        if isinstance(base_w, (list, tuple)) else base_w)
            if isinstance(scaled_w, torch.Tensor): scaled_w = scaled_w * alpha
            else: scaled_w = float(scaled_w) * alpha
            
            # 2. 샘플별 동적 가중치 (이벤트 기반)
            sample_w = self._get_dynamic_weight(k, w_cond_batch, high_weight_value=5.0)

            # 3. [핵심] Anomaly Mask 계산
            # 학습 모드일 때만 마스킹 적용 (검증 시에는 모든 데이터로 평가하는 것이 일반적이나, 
            # 논문에 따르면 학습 loss에 적용하는 것이므로 여기선 train/val 구분 없이 로직상 적용됨.
            # 하지만 보통 val metric은 마스킹 없이 순수 오차를 보는 게 맞음. 필요 시 분기 처리)
            if self.model.training: 
                anomaly_mask = self._compute_anomaly_mask(yh.detach(), y_lb, yt) # yh는 gradient 끊고 점수 계산
            else:
                anomaly_mask = torch.ones_like(yh)

            # 4. 최종 가중치 결합: (기본) * (이벤트) * (마스크)
            # anomaly_mask: (B, D), sample_w: (B,) -> (B,1)
            final_w = scaled_w * sample_w.unsqueeze(-1) * anomaly_mask
            
            losses.append(self._weighted_loss(yh, yt, final_w, self.device))
        return losses

    def step(self, batch, train=True):
        xs_list, lens_list, ys_list, ys_future_list, xs_future_list, w_cond_batch, w_future_batch = self._to_device(batch)
        
        self.model.train(train)
        self.opt.zero_grad(set_to_none=True)
        max_stage_idx = (max(self.active_stages) if self.active_stages else None)
        
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # 1. 메인 모델 예측
            y_hats, _ = self.model(xs_list, lens_list, max_stage=max_stage_idx)
            
            # 2. [추가] Lower Bound Model 예측 (Gradient 불필요)
            with torch.no_grad():
                y_hats_lb, _ = self.lb_model(xs_list, lens_list, max_stage=max_stage_idx)
            
            # 3. 1-step Loss 계산 (여기서 마스킹 수행)
            one_step_losses = self._one_step_losses(y_hats, y_hats_lb, ys_list, w_cond_batch)
            one_step_total = sum(one_step_losses) if len(one_step_losses) > 0 else torch.tensor(0.0, device=self.device)
            
            # 4. Rollout Loss (여기엔 마스킹 미적용 - 비용 문제 및 간접 효과 기대)
            if self.rollout_horizon > 0 and self.feedback_adapter is not None:
                xs_copy = [[x.clone() for x in stage] for stage in xs_list]
                rollout_total = self._rollout_losses_stagewise(
                    xs_copy, lens_list, ys_future_list, xs_future_list, w_future_batch, max_stage_idx
                )
                total = one_step_total + self.rollout_weight * rollout_total
            else:
                total = one_step_total
        
        if train:
            self.scaler.scale(total).backward()
            if self.grad_clip is not None:
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            # ReduceLROnPlateau는 step마다 호출하지 않음
            
        logs = {"total": float(total.detach()), "one_step": float(one_step_total.detach())}
        for k, lk in enumerate(one_step_losses):
            logs[f"loss_stage{k+1}"] = float(lk.detach())
        return logs

    def fit(self, train_loader, val_loader=None, epochs=200, log_every=50,
            early_stop_patience=20, ckpt_path="best.ckpt"):
        """
        [BAS 핵심 주석] Phase-based Curriculum Learning
        화학 공정은 앞단의 반응기(Upstream)가 불안정하면 뒷단의 배기가스(Downstream)도
        필연적으로 불안정합니다. 따라서 본 학습 로직은 다음 순서로 모델을 튜닝합니다.
        
        1. Phase '1단온도': 반응기 앞부분만 집중 학습 (나머지 가중치 0)
        2. Phase '3단온도': 1단 고정 + 3단 집중 학습
        ...
        이런 식으로 공정의 흐름을 따라 순차적으로 파라미터를 최적화(Freezing/Unfreezing)합니다.
        이는 실제 공장의 시운전(Commissioning) 절차와 유사합니다.
        """
        self.st_time = time.time()
        self.last_time = time.time()
        best = float("inf")
        patience = 0

        # self.phase = '1단온도_1'
        # self.set_active_stages({0})
        # self.set_weight_scale([1.0, 0, 0, 0])
        # for k in [0]:
        #     set_stage_trainable(self.model, k, True)

        self.phase = 'TMS_2'
        self.set_active_stages({0,1,2,3})
        self.set_weight_scale([1.0, 1.0, 1.0, 1.0])
        for k in [0,1,2,3]:
            set_stage_trainable(self.model, k, True)
        
        print('phase changed', self.phase)
        logger.info('phase changed {}'.format(self.phase))

        
        for ep in range(1, epochs+1):
            # ---- train ----
            agg = {}
            for it, batch in enumerate(train_loader, 1):
                log = self.step(batch, train=True)
                for k,v in log.items(): agg[k] = agg.get(k,0.0) + v
                if it % log_every == 0:
                    msg = " ".join([f"{k}:{v/it:.4f}" for k,v in agg.items()])
                    print(f"[E{ep} I{it}] {msg} ({round(time.time()-self.last_time,2)})")
                    logger.info(f"[E{ep} I{it}] {msg} ({round(time.time()-self.last_time,2)})")
                    self.last_time = time.time()
            
                    # ---- val (Epoch 단위) ----
                    val_metric = None
                    if val_loader is not None:
                        agg = {}
                        with torch.no_grad():
                            for it, batch in enumerate(val_loader, 1):
                                log = self.step(batch, train=False) # train=False면 마스킹 없이 순수 Loss 계산
                                for k,v in log.items(): agg[k] = agg.get(k,0.0) + v
                        val_metric = agg["total"]/it
                        
                        print(f"[E{ep} VAL] Total: {val_metric:.4f}")
        
                        if isinstance(self.sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.sched.step(val_metric)
                        
                        # Early stopping & best ckpt
                        if val_metric < best:
                            best = val_metric
                            patience = 0
                            torch.save(self.model.state_dict(), ckpt_path)
                            
                            # [핵심] New Best 달성 시 -> Lower Bound Model 업데이트 (Target Network Update)
                            self._update_lb_model()
                            
                            print(f"[E{ep}] ✅ New best: {best:.6f} (LB Model Updated) (saved {ckpt_path})")
                            logger.info(f"[E{ep}] ✅ New best: {best:.6f} (LB Model Updated)")
                        else:
                            patience += 1
                            if patience >= early_stop_patience:
                                print(f"⛳ Early stopped at epoch {ep}. Best={best:.6f}")
                                self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
                                break
                            
                            # Phase change logic based on patience
                            # [BAS 설명] 학습 정체(Plateau) 시 다음 공정 단계(Stage)를 해금(Unlock)하는 로직
                            elif patience >= early_stop_patience/2:
                                if self.phase == '1단온도_1':
                                    self.set_active_stages({0,1})
                                    self.set_weight_scale([1.0, 0.3, 0.0, 0.0])
                                    set_stage_trainable(self.model, 1, True)
                                    best = float("inf")
                                    sd = torch.load(ckpt_path, map_location=self.device)
                                    self.model.load_state_dict(sd, strict=True)
        
                                    self.phase = '3단온도_1'
                                    self.opt = torch.optim.AdamW(
                                            filter(lambda p: p.requires_grad, self.model.parameters()),
                                            lr=self.lr, weight_decay=self.wd
                                        )
                                    self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                        self.opt, mode="min", factor=0.5, patience=30, min_lr=1e-8
                                    )
                                    self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
                                    self.opt.zero_grad(set_to_none=True)
                                    logger.info('phase changed {}'.format(self.phase))
                                elif self.phase == '3단온도_1':
                                # if self.phase == '1단온도_1':
                                    self.set_active_stages({0,1})
                                    self.set_weight_scale([1.0, 1.0, 0.0, 0.0])
                                    set_stage_trainable(self.model, 1, True)
                                    best = float("inf")
                                    sd = torch.load(ckpt_path, map_location=self.device)
                                    self.model.load_state_dict(sd, strict=True)
                                    self.phase = '3단온도_2'
                                    self.opt = torch.optim.AdamW(
                                            filter(lambda p: p.requires_grad, self.model.parameters()),
                                            lr=self.lr, weight_decay=self.wd
                                        )
                                    self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                        self.opt, mode="min", factor=0.5, patience=30, min_lr=1e-8
                                    )
                                    self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
                                    self.opt.zero_grad(set_to_none=True)
                                    logger.info('phase changed {}'.format(self.phase))
                                elif self.phase == '3단온도_2':
                                    self.set_active_stages({0,1,2})
                                    self.set_weight_scale([1.0, 1.0, 0.3, 0.0])
                                    set_stage_trainable(self.model, 2, True)
                                    best = float("inf")
                                    sd = torch.load(ckpt_path, map_location=self.device)
                                    self.model.load_state_dict(sd, strict=True)
                                    self.phase = 'Analyzer_1'
                                    self.opt = torch.optim.AdamW(
                                            filter(lambda p: p.requires_grad, self.model.parameters()),
                                            lr=self.lr, weight_decay=self.wd
                                        )
                                    self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                        self.opt, mode="min", factor=0.5, patience=30, min_lr=1e-8
                                    )
                                    self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
                                    self.opt.zero_grad(set_to_none=True)
                                    logger.info('phase changed {}'.format(self.phase))
                                elif self.phase == 'Analyzer_1':
                                # elif self.phase == '3단온도_2':
                                    self.set_active_stages({0,1,2})
                                    self.set_weight_scale([1.0, 1.0, 1.0, 0.0])
                                    set_stage_trainable(self.model, 2, True)
                                    best = float("inf")
                                    sd = torch.load(ckpt_path, map_location=self.device)
                                    self.model.load_state_dict(sd, strict=True)
                                    self.phase = 'Analyzer_2'
                                    self.opt = torch.optim.AdamW(
                                            filter(lambda p: p.requires_grad, self.model.parameters()),
                                            lr=self.lr, weight_decay=self.wd
                                        )
                                    self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                        self.opt, mode="min", factor=0.5, patience=30, min_lr=1e-8
                                    )
                                    self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
                                    self.opt.zero_grad(set_to_none=True)
                                    logger.info('phase changed {}'.format(self.phase))
                                elif self.phase == 'Analyzer_2':
                                    self.set_active_stages({0,1,2,3})
                                    self.set_weight_scale([1.0, 1.0, 1.0, 0.3])
                                    for k in [0,1,2,3]: 
                                        set_stage_trainable(self.model, k, True)
                                    best = float("inf")
                                    sd = torch.load(ckpt_path, map_location=self.device)
                                    self.model.load_state_dict(sd, strict=True)
                                    self.phase = 'TMS_1'
                                    self.opt = torch.optim.AdamW(
                                            filter(lambda p: p.requires_grad, self.model.parameters()),
                                            lr=self.lr, weight_decay=self.wd
                                        )
                                    self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                        self.opt, mode="min", factor=0.5, patience=30, min_lr=1e-8
                                    )
                                    self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
                                    self.opt.zero_grad(set_to_none=True)
                                    logger.info('phase changed {}'.format(self.phase))
                                elif self.phase == 'TMS_1':
                                # elif self.phase == 'Analyzer_2':
                                    self.set_active_stages({0,1,2,3})
                                    self.set_weight_scale([1.0, 1.0, 1.0, 1.0])
                                    for k in [0,1,2,3]: 
                                        set_stage_trainable(self.model, k, True)
                                    best = float("inf")
                                    sd = torch.load(ckpt_path, map_location=self.device)
                                    self.model.load_state_dict(sd, strict=True)
                                    self.phase = 'TMS_2'
                                    self.opt = torch.optim.AdamW(
                                            filter(lambda p: p.requires_grad, self.model.parameters()),
                                            lr=self.lr, weight_decay=self.wd
                                        )
                                    self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                        self.opt, mode="min", factor=0.5, patience=30, min_lr=1e-8
                                    )
                                    self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
                                    self.opt.zero_grad(set_to_none=True)
                                    logger.info('phase changed {}'.format(self.phase))
                            

def set_stage_trainable(model: MultiStageProcessModel, stage_idx: int, on: bool = True):
    # stream_grus, stream_projs, stage_fusions, xattn_blocks, heads
    for m in [model.stream_grus[stage_idx], model.stream_projs[stage_idx],
              model.stage_fusions[stage_idx], model.heads[stage_idx]]:
        for p in m.parameters():
            p.requires_grad = on
    if stage_idx < len(model.xattn_blocks) and not isinstance(model.xattn_blocks[stage_idx], nn.Identity):
        for p in model.xattn_blocks[stage_idx].parameters():
            p.requires_grad = on

def load_weights_with_dropout_insertion(model, ckpt_path, device):
    """
    [버그 수정판]
    단순 문자열 치환이 아닌, 구조를 파싱하여 정확히 'heads' 내부의 'Linear' 레이어 인덱스만
    3 -> 4로 변경합니다. Stage 인덱스와 혼동하지 않습니다.
    [BAS 설명] 모델 버전 관리용 유틸리티
    이전 버전 모델(구조가 달랐던 경우)의 가중치를 현재 모델 구조에 맞게
    자동으로 매핑(Mapping)하여 불러오는 기능을 수행합니다.
    """
    old_state_dict = torch.load(ckpt_path, map_location=device)
    new_state_dict = model.state_dict()
    converted_state_dict = {}
    print("[Weight Transfer V2] 정밀 파라미터 이식 시작...")

    for key, value in old_state_dict.items():
        parts = key.split('.')
        if parts[0] == 'heads':
            stage_idx = parts[1]
            layer_idx = parts[2]
            if layer_idx == '3':
                parts[2] = '4'
                new_key = ".".join(parts)
                if new_key in new_state_dict:
                    converted_state_dict[new_key] = value
                else:
                    print(f" [Error] {new_key} should exist but not found.")
            else:
                if key in new_state_dict:
                    converted_state_dict[key] = value
        else:
            if key in new_state_dict:
                converted_state_dict[key] = value

    model.load_state_dict(converted_state_dict, strict=False)
    print("[Weight Transfer V2] 완료! 모든 스테이지가 정상 적용되었습니다.")