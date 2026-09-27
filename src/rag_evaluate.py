"""
RAG效果评估脚本 rag_evaluate.py
================================
用途：输入测试问答对，计算召回率和命中率，输出评估报告
存放路径：E:/novel_agent/src/rag_evaluate.py

评估指标：
1. 关键词召回率(recall@k)：预期关键词是否出现在Top-K检索结果中
2. chunk命中率(hit_rate@k)：预期chunk_id是否在Top-K检索结果中
3. 平均倒数排名(MRR)：预期结果的倒数排名

使用方式：
  1. 准备测试集JSON文件（格式见 SAMPLE_TESTSET）
  2. 运行：python rag_evaluate.py --testset testset.json
  3. 查看评估报告

测试集格式：
  [
    {
      "question": "唐三的武魂是什么",
      "expected_keywords": ["蓝银草", "蓝银皇"],
      "expected_chunk_ids": [100, 200]   // 可选，已知的正确chunk_id
    },
    ...
  ]
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import argparse
import time
from pathlib import Path

from config import TOP_K_FINAL, TOP_K_VECTOR, TOP_K_BM25
from utils import logger

# 默认测试集（斗罗大陆已知问答）
SAMPLE_TESTSET = [
    {
        "question": "唐三的武魂是什么",
        "expected_keywords": ["蓝银草", "蓝银皇", "武魂"],
        "expected_chunk_ids": []
    },
    {
        "question": "小舞是谁",
        "expected_keywords": ["小舞", "十万年", "魂兽", "柔骨魅兔"],
        "expected_chunk_ids": []
    },
    {
        "question": "史莱克学院在哪里",
        "expected_keywords": ["史莱克", "学院", "蓝霸"],
        "expected_chunk_ids": []
    },
    {
        "question": "唐三的父亲是谁",
        "expected_keywords": ["唐昊", "父亲", "昊天"],
        "expected_chunk_ids": []
    },
    {
        "question": "大师的武魂是什么",
        "expected_keywords": ["大师", "罗三炮", "变异"],
        "expected_chunk_ids": []
    },
]


def evaluate_recall_at_k(retrieved_docs, expected_keywords, k=5):
    """计算关键词召回率：Top-K检索结果中包含多少预期关键词

    返回：命中的关键词数 / 预期关键词总数
    """
    if not expected_keywords:
        return 1.0  # 无预期关键词，视为全部命中

    top_k_docs = retrieved_docs[:k]
    # 把Top-K片段的内容拼成一个长文本
    combined_text = " ".join([d["page_content"] for d, _ in top_k_docs])

    hit_keywords = [kw for kw in expected_keywords if kw in combined_text]
    recall = len(hit_keywords) / len(expected_keywords)
    return recall, hit_keywords


def evaluate_hit_rate_at_k(retrieved_docs, expected_chunk_ids, k=5):
    """计算chunk命中率：预期chunk_id是否出现在Top-K中

    返回：命中的chunk数 / 预期chunk总数
    """
    if not expected_chunk_ids:
        return 1.0, []  # 无预期chunk_id，视为全部命中

    top_k_docs = retrieved_docs[:k]
    retrieved_cids = [d["metadata"].get("chunk_id") for d, _ in top_k_docs]
    hit_cids = [cid for cid in expected_chunk_ids if cid in retrieved_cids]
    hit_rate = len(hit_cids) / len(expected_chunk_ids)
    return hit_rate, hit_cids


def evaluate_mrr(retrieved_docs, expected_chunk_ids):
    """计算平均倒数排名(MRR)：预期chunk_id在检索结果中的倒数排名

    如果预期chunk_id在第1位，MRR贡献=1/1=1.0
    如果在第3位，MRR贡献=1/3=0.33
    未命中贡献=0
    """
    if not expected_chunk_ids:
        return 0.0

    for rank, (doc, _) in enumerate(retrieved_docs, 1):
        cid = doc["metadata"].get("chunk_id")
        if cid in expected_chunk_ids:
            return 1.0 / rank
    return 0.0


def run_evaluation(testset, top_k=TOP_K_FINAL):
    """运行RAG评估

    对每个测试问题：
    1. 调用混合检索获取Top-K结果
    2. 计算关键词召回率、chunk命中率、MRR
    3. 汇总输出评估报告
    """
    # 延迟导入检索器（避免首次import就加载模型）
    from retrieval_tools import get_retriever
    retriever = get_retriever()

    results = []
    total_recall = 0.0
    total_hit_rate = 0.0
    total_mrr = 0.0

    logger.info(f"[评估] 开始评估，共 {len(testset)} 个测试问题，Top-K={top_k}")
    logger.info("=" * 70)

    for i, item in enumerate(testset):
        question = item["question"]
        expected_keywords = item.get("expected_keywords", [])
        expected_chunk_ids = item.get("expected_chunk_ids", [])

        logger.info(f"\n[问题{i+1}] {question}")
        logger.info(f"  预期关键词: {expected_keywords}")

        try:
            # 调用混合检索
            retrieved = retriever.retrieve(question, top_k=top_k)
            logger.info(f"  检索到 {len(retrieved)} 个片段")

            # 计算指标
            recall, hit_kws = evaluate_recall_at_k(retrieved, expected_keywords, k=top_k)
            hit_rate, hit_cids = evaluate_hit_rate_at_k(retrieved, expected_chunk_ids, k=top_k)
            mrr = evaluate_mrr(retrieved, expected_chunk_ids)

            total_recall += recall
            total_hit_rate += hit_rate
            total_mrr += mrr

            logger.info(f"  关键词召回率: {recall:.2%} (命中: {hit_kws})")
            logger.info(f"  chunk命中率: {hit_rate:.2%}")
            logger.info(f"  MRR: {mrr:.4f}")

            # 展示Top-3检索结果
            for j, (doc, score) in enumerate(retrieved[:3]):
                cid = doc["metadata"].get("chunk_id")
                logger.info(f"    Top{j+1} (chunk_id={cid}, score={score:.4f}): {doc['page_content'][:60]}...")

            results.append({
                "question": question,
                "recall": round(recall, 4),
                "hit_rate": round(hit_rate, 4),
                "mrr": round(mrr, 4),
                "hit_keywords": hit_kws,
                "retrieved_count": len(retrieved),
            })
        except Exception as e:
            logger.error(f"  评估失败: {e}")
            results.append({
                "question": question,
                "recall": 0.0,
                "hit_rate": 0.0,
                "mrr": 0.0,
                "error": str(e),
            })

    # 汇总
    n = len(testset)
    avg_recall = total_recall / n if n > 0 else 0
    avg_hit_rate = total_hit_rate / n if n > 0 else 0
    avg_mrr = total_mrr / n if n > 0 else 0

    logger.info("\n" + "=" * 70)
    logger.info("[评估报告] RAG效果评估汇总")
    logger.info("=" * 70)
    logger.info(f"  测试问题数: {n}")
    logger.info(f"  平均关键词召回率: {avg_recall:.2%}")
    logger.info(f"  平均chunk命中率: {avg_hit_rate:.2%}")
    logger.info(f"  平均MRR: {avg_mrr:.4f}")
    logger.info("=" * 70)

    # 输出详细结果表
    logger.info("\n[详细结果]")
    logger.info(f"{'问题':<30} {'召回率':<10} {'命中率':<10} {'MRR':<10}")
    logger.info("-" * 60)
    for r in results:
        q = r["question"][:28]
        logger.info(f"{q:<30} {r['recall']:<10.2%} {r['hit_rate']:<10.2%} {r['mrr']:<10.4f}")

    return {
        "summary": {
            "test_count": n,
            "avg_recall": round(avg_recall, 4),
            "avg_hit_rate": round(avg_hit_rate, 4),
            "avg_mrr": round(avg_mrr, 4),
        },
        "details": results,
    }


def main():
    """主入口：解析参数，加载测试集，运行评估"""
    parser = argparse.ArgumentParser(description="RAG效果评估脚本")
    parser.add_argument("--testset", type=str, default=None,
                        help="测试集JSON文件路径（不指定则用内置样例）")
    parser.add_argument("--top_k", type=int, default=TOP_K_FINAL,
                        help=f"评估Top-K（默认{TOP_K_FINAL}）")
    parser.add_argument("--output", type=str, default=None,
                        help="评估报告输出路径（JSON格式）")
    args = parser.parse_args()

    # 加载测试集
    if args.testset and os.path.exists(args.testset):
        with open(args.testset, "r", encoding="utf-8") as f:
            testset = json.load(f)
        logger.info(f"[评估] 加载测试集: {args.testset} ({len(testset)}条)")
    else:
        logger.info("[评估] 使用内置测试集")
        testset = SAMPLE_TESTSET

    # 运行评估
    report = run_evaluation(testset, top_k=args.top_k)

    # 输出报告到文件
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"[评估] 报告已保存到: {args.output}")


if __name__ == "__main__":
    main()
