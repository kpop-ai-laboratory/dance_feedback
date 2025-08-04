# graph_utils.py

import matplotlib.pyplot as plt
import os

def generate_sequence_score_bar(scores, save_path="sequence_score_bar.png"):
    """
    시퀀스별 점수 바 그래프 생성 후 PNG로 저장
    :param scores: List[int] - 시퀀스 점수들 (0~100)
    :param save_path: 저장할 이미지 경로
    :return: 저장된 이미지 경로
    """
    plt.figure(figsize=(12, 2))
    bar_colors = ["#4CAF50" if s >= 80 else "#FFC107" if s >= 60 else "#F44336" for s in scores]
    plt.bar(range(len(scores)), scores, color=bar_colors)
    plt.xticks(range(len(scores)), [f"S{i+1}" for i in range(len(scores))], fontsize=8)
    plt.yticks([0, 25, 50, 75, 100], fontsize=8)
    plt.ylim(0, 105)
    plt.xlabel("Sequence")
    plt.ylabel("Score")
    plt.title("Sequence-wise Similarity Score", fontsize=10)
    plt.tight_layout()

    # 이미지 저장
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


if __name__ == '__main__':
    dummy_scores = [85, 78, 92, 60, 70, 88, 55, 95]
    path = generate_sequence_score_bar(dummy_scores)
    print(f"✅ 그래프 저장 완료: {path}")