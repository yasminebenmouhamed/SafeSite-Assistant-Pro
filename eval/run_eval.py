"""Transparent evaluation: retrieval hit@k, rough answer inclusion, refusal, latency.
Usage: python -m eval.run_eval --mode hybrid --output eval/results.local.json
Metrics are deliberately simple and not equivalent to human correctness review.
"""
import argparse
import json
import re
import statistics
from pathlib import Path
from safesite.config import Settings
from safesite.engine import RAGEngine, REFUSAL


def norm(x):
    return ' '.join(re.findall(r'\w+', x.casefold()))


def evaluate(engine, dataset, mode):
    rows = []
    for item in dataset:
        try:
            result = engine.ask(item['question'], mode=mode)
            answer = result['answer']
            refused = REFUSAL.casefold() in answer.casefold()
            retrieval_hit = any(s['file'] == item['source'] for s in result['sources']) if item['source'] else None
            # Heuristic only: expected exact key phrase in generated answer.
            answer_match = norm(item['expected']) in norm(answer) if not item['unanswerable'] else refused
            rows.append({'question': item['question'], 'expected': item['expected'], 'actual': answer,
                         'retrieval_hit': retrieval_hit, 'answer_match_heuristic': answer_match,
                         'correct_refusal': refused if item['unanswerable'] else None,
                         'latency_ms': result['latency_ms'], 'error': None})
        except Exception as exc:
            rows.append({'question': item['question'], 'error': type(exc).__name__, 'retrieval_hit': None,
                         'answer_match_heuristic': False, 'correct_refusal': None, 'latency_ms': None})
    applicable = [r['retrieval_hit'] for r in rows if r['retrieval_hit'] is not None]
    refusals = [r['correct_refusal'] for r in rows if r['correct_refusal'] is not None]
    latencies = [r['latency_ms'] for r in rows if r['latency_ms'] is not None]
    return {
        'mode': mode, 'cases': len(rows),
        'retrieval_hit_at_k': sum(applicable) / len(applicable) if applicable else None,
        'answer_inclusion_heuristic': sum(r['answer_match_heuristic'] for r in rows) / len(rows) if rows else None,
        'correct_refusals': sum(refusals), 'total_unanswerable': len(refusals),
        'average_latency_ms': round(statistics.mean(latencies)) if latencies else None,
        'note': 'These are proxy metrics, not manually verified factual accuracy. Compare all mode results and review errors.',
        'details': rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['similarity', 'mmr', 'hybrid'], default='hybrid')
    parser.add_argument('--output', default='eval/results.local.json')
    args = parser.parse_args()
    questions = json.loads(Path('eval/questions.json').read_text(encoding='utf-8'))
    output = evaluate(RAGEngine(Settings()), questions, args.mode)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k != 'details'}, indent=2))

if __name__ == '__main__':
    main()
