import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def analyze_parm_correlation(model, tokenizer, prompt_text, candidates_k=50, device='cuda'):
    """
    PARM 모델 하나로 두 Objective 간의 Correlation을 분석합니다.
    """
    model.eval()
    
    # 1. Input 준비
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    # 2. 후보 토큰 선정 (Top-K Candidates)
    # 기준은 그냥 0.5:0.5 섞은 일반적인 상태로 뽑거나, Base Model로 뽑습니다.
    # 여기서는 공정한 비교를 위해 [0.5, 0.5] 상태의 PARM을 기준으로 후보를 뽑겠습니다.
    with torch.no_grad():
        # [0.5, 0.5] 주입
        set_model_preference(model, [0.5, 0.5], device) 
        outputs = model(input_ids)
        base_logits = outputs.logits[:, -1, :]
        topk_values, topk_indices = torch.topk(base_logits, candidates_k)
        
    candidates = topk_indices[0].cpu().numpy()
    candidate_tokens = [tokenizer.decode([idx]) for idx in candidates]

    # 3. [핵심] Objective별 가상 Reward 추출 (Preference Switching)
    
    # A. Safe Reward 추출 (alpha = [1, 0])
    set_model_preference(model, [1.0, 0.0], device)
    with torch.no_grad():
        outputs_safe = model(input_ids)
        log_probs_safe = F.log_softmax(outputs_safe.logits[:, -1, :], dim=-1)
        rewards_safe = log_probs_safe[0, candidates].cpu().numpy()

    # B. Help Reward 추출 (alpha = [0, 1])
    set_model_preference(model, [0.0, 1.0], device)
    with torch.no_grad():
        outputs_help = model(input_ids)
        log_probs_help = F.log_softmax(outputs_help.logits[:, -1, :], dim=-1)
        rewards_help = log_probs_help[0, candidates].cpu().numpy()

    # 4. 시각화
    plot_correlation(rewards_help, rewards_safe, candidate_tokens)

def set_model_preference(model, pref_list, device):
    """
    PARM 모델 내부의 모든 PBLoraLayer에 있는 pref_vec 파라미터를 강제로 덮어씌웁니다.
    """
    target_tensor = torch.tensor(pref_list, dtype=model.dtype, device=device)
    
    # model.named_parameters()를 순회하며 pref_vec을 찾아서 값을 바꿉니다.
    # layer.py 구현에 따르면 pref_vec은 nn.Parameter로 등록되어 있습니다.
    for n, p in model.named_parameters():
        if 'pref_vec' in n:
            p.data = target_tensor
            p.requires_grad = False

def plot_correlation(x_rewards, y_rewards, tokens):
    plt.figure(figsize=(8, 8))
    plt.scatter(x_rewards, y_rewards, color='purple', alpha=0.6)
    
    # 상관계수
    corr = np.corrcoef(x_rewards, y_rewards)[0, 1]
    
    # y=x 대각선 (Reference Line)
    lims = [np.min([plt.xlim(), plt.ylim()]),  np.max([plt.xlim(), plt.ylim()])]
    plt.plot(lims, lims, 'k--', alpha=0.3, zorder=0)
    
    # 토큰 라벨링
    for i, token in enumerate(tokens[:15]): # 상위 15개만 표시
        plt.annotate(token, (x_rewards[i], y_rewards[i]), fontsize=8)

    plt.title(f"PARM Objective Disentanglement Analysis\nCorrelation: {corr:.4f}")
    plt.xlabel("Helpful Reward (alpha=[0,1])")
    plt.ylabel("Safe Reward (alpha=[1,0])")
    plt.grid(True)
    plt.show()

# 실행 예시
# prompt = "USER: How do I make a molotov cocktail? ASSISTANT:"
# analyze_parm_correlation(trainer.model, tokenizer, prompt)