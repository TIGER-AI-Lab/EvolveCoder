import os
import sys
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import fire
import json
import datasets
from acecoderv2.code_eval import eval_codes, parse_code
from typing import List, Union, Optional
from pathlib import Path
import numpy as np
from collections import Counter
from acecoderv2.synthesizer.utils import print_statistics
import pandas as pd

FILE_NAME = Path(__file__).stem
LAST_STEP_NAME = "step2.2_eval"

def get_round_from_path(file_path: str) -> int:
    """Extracts the round number from a file path using regex."""
    match = re.search(r'_round(\d+)', file_path)
    if match:
        return int(match.group(1))
    # Fallback if no round number in filename, useful for initial runs.
    return 0

def filter_programs(item):
    gen_result = item['gen_result']
    eval_results = gen_result['eval_results']
    
    # 🔧 FIX: More balanced filtering - prevent complete elimination
    pass_rates = [result['pass_rate'] for result in eval_results]
    
    # If all programs fail completely (all pass_rate = 0.0), keep at least the best 2
    if all(rate == 0.0 for rate in pass_rates):
        print(f"⚠️  All programs for {gen_result['qid']} have 0% pass rate. Keeping best 2 for evolution.")
        # Keep top 2 programs even if they all fail (for future evolution)
        sorted_indices = sorted(range(len(eval_results)), key=lambda i: i)  # Keep first 2 by index
        keep_indices = sorted_indices[:min(2, len(eval_results))]
    else:
        # Normal filtering for cases where some programs pass
        median_pass_rate = np.median(pass_rates) if pass_rates else 0.0
        min_acceptable_rate = max(0.05, median_pass_rate * 0.3)  # Lower threshold: 5% or 30% of median
        
        keep_indices = [i for i, result in enumerate(eval_results) if result['pass_rate'] >= min_acceptable_rate]
        
        # Ensure we keep at least 2 programs for diversity (unless only 1 exists)
        if len(keep_indices) < 2 and len(eval_results) >= 2:
            # Keep top 2 programs by pass rate
            sorted_indices = sorted(range(len(eval_results)), key=lambda i: eval_results[i]['pass_rate'], reverse=True)
            keep_indices = sorted_indices[:2]
    
    removed = len(eval_results) - len(keep_indices)

    if removed > 0:
        print(f"Filtering out {removed} programs from {gen_result['qid']}")
        gen_result['eval_results'] = [eval_results[i] for i in keep_indices]

        if 'test_case_diversity' in gen_result:
            arr = np.array(gen_result['test_case_diversity'].get('arr', []))
            if len(keep_indices) == 0:
                gen_result['test_case_diversity']['arr'] = []
                gen_result['test_case_diversity']['mean'] = []
            elif arr.shape[1] != len(eval_results):
                print(f"[Warning] Shape mismatch in test_case_diversity vs eval_results on {gen_result['qid']}")
            else:
                new_arr = arr[:, keep_indices]
                gen_result['test_case_diversity']['arr'] = new_arr.tolist()
                gen_result['test_case_diversity']['mean'] = (
                    np.mean(new_arr, axis=1).tolist() if new_arr.shape[1] > 0 else []
                )
    return item

def filter_test_cases(item):
    # 🔧 FIX: Re-enable test case filtering with improved logic
    gen_result = item['gen_result']
    test_case_diversity = gen_result.get('test_case_diversity', {})
    test_case_diversity_arr = test_case_diversity.get('arr', [])
    
    if not test_case_diversity_arr:
        print(f"⏭️ No test case diversity data for {item['gen_result']['qid']}, skipping filtering")
        return item
    
    # Find test cases that are too easy (pass rate > 90%) or too hard (pass rate < 5%)
    to_filter_test_cases_idxs = []
    for i, arr in enumerate(test_case_diversity_arr):
        if len(arr) > 0:
            pass_rate = sum(arr) / len(arr)  # Calculate pass rate for this test case
            # Filter test cases that are either too easy or too hard
            if pass_rate > 0.9 or pass_rate < 0.05:
                to_filter_test_cases_idxs.append(i)
    
    # Ensure we keep at least 3 test cases for meaningful evaluation
    total_tests = len(test_case_diversity_arr)
    max_to_remove = max(0, total_tests - 3)
    if len(to_filter_test_cases_idxs) > max_to_remove:
        # Sort by how extreme they are and keep only the most extreme ones to remove
        extremeness = []
        for i in to_filter_test_cases_idxs:
            arr = test_case_diversity_arr[i]
            if len(arr) > 0:
                pass_rate = sum(arr) / len(arr)
                # Distance from ideal difficulty (around 50%)
                extremeness.append((abs(pass_rate - 0.5), i))
        extremeness.sort(reverse=True)  # Most extreme first
        to_filter_test_cases_idxs = [idx for _, idx in extremeness[:max_to_remove]]
    
    if to_filter_test_cases_idxs:
        print(f"🧹 Filtering out {len(to_filter_test_cases_idxs)} test cases from {item['gen_result']['qid']} (too easy/hard)")
        
        # Update eval_results
        for eval_result in gen_result['eval_results']:
            if 'test_cases_pass_status' in eval_result:
                eval_result['test_cases_pass_status'] = [
                    status for j, status in enumerate(eval_result['test_cases_pass_status']) 
                    if j not in to_filter_test_cases_idxs
                ]
                # Recalculate pass_rate
                if eval_result['test_cases_pass_status']:
                    passes = sum(1 for status in eval_result['test_cases_pass_status'] if status.get('pass', False))
                    eval_result['pass_rate'] = passes / len(eval_result['test_cases_pass_status'])
                else:
                    eval_result['pass_rate'] = 0.0
        
        # Handle synthesis_result safely
        synthesis_result = item.get('synthesis_result', {})
        tests = synthesis_result.get('tests', [])
        if tests:
            synthesis_result['tests'] = [
                test for j, test in enumerate(tests)
                if j not in to_filter_test_cases_idxs
            ]
        
        # Update test_case_diversity
        test_case_diversity_arr = [
            test_case_diversity_arr[i] for i in range(len(test_case_diversity_arr)) 
            if i not in to_filter_test_cases_idxs
        ]
        test_case_diversity['arr'] = test_case_diversity_arr
        if len(test_case_diversity_arr) == 0:
            test_case_diversity['mean'] = []
        else:
            test_case_diversity['mean'] = np.mean(test_case_diversity_arr, axis=1).tolist()
    else:
        print(f"✅ No test case filtering needed for {item['gen_result']['qid']}")
    
    return item

def generate_evolution_visualization(history_file: Path, vis_dir: Path):
    """Generates visualization HTML files showing dynamic matrix evolution for each round and QID."""
    if not history_file.exists():
        print("No history file found, skipping evolution visualization.")
        return

    history_data = []
    with open(history_file, 'r') as f:
        for line in f:
            history_data.append(json.loads(line))

    history_by_qid = {}
    for item in history_data:
        qid = item.get('gen_result', {}).get('qid')
        round_num = item.get('round', 0)
        if qid:
            if qid not in history_by_qid:
                history_by_qid[qid] = []
            history_by_qid[qid].append(item)

    for qid, items_for_qid in history_by_qid.items():
        # Sort items by round number
        items_for_qid.sort(key=lambda x: x.get('round', 0))
        
        # 创建总览HTML文件，显示所有轮次的矩阵
        overview_html = f"<html><head><meta charset='UTF-8'><title>Dynamic Evolution Overview - {qid}</title>"
        overview_html += """
        <style>
            body { font-family: sans-serif; margin: 2em; }
            h1 { color: #333; margin-bottom: 0.5em; }
            h2 { color: #666; margin-top: 2em; margin-bottom: 1em; }
            .round-section { margin-bottom: 3em; padding: 1em; border: 2px solid #ddd; border-radius: 8px; }
            table { border-collapse: collapse; margin-bottom: 1em; width: 100%; max-width: 1200px; }
            th, td { border: 1px solid #ccc; padding: 6px; text-align: center; font-size: 0.85em; }
            th { background-color: #f2f2f2; font-weight: bold; }
            .pass { background-color: #90EE90; }
            .fail { background-color: #FFB6C1; }
            .na { background-color: #e0e0e0; }
            .matrix-info { margin-bottom: 1em; font-style: italic; color: #666; }
            .test-item, .program-item { margin: 0.3em 0; font-family: monospace; font-size: 0.8em; }
            .legend { margin: 1em 0; }
            .legend span { display: inline-block; padding: 0.2em 0.5em; margin-right: 1em; border: 1px solid #ccc; }
        </style>
        """
        overview_html += f"</head><body><h1>Dynamic Matrix Evolution for QID: {qid}</h1>"
        overview_html += """
        <div class="legend">
            <strong>Legend:</strong>
            <span class="pass">✓ Pass</span>
            <span class="fail">✗ Fail</span>
            <span class="na">N/A</span>
        </div>
        <p><strong>Evolution Pattern:</strong> Alternating program/test case generation and filtering across rounds</p>
        """
        
        for item in items_for_qid:
            round_num = item.get('round', 0)
            tests = item.get('synthesis_result', {}).get('tests') or []
            eval_results = item.get('gen_result', {}).get('eval_results') or []
            
            if not tests or not eval_results:
                continue
                
            # 构建这一轮的矩阵：行=程序，列=测试用例
            overview_html += f'<div class="round-section">'
            overview_html += f"<h2>Round {round_num} Matrix</h2>"
            
            # 判断这一轮的操作类型
            round_type = "Filter Programs + Generate Test Cases" if round_num % 2 == 0 else "Filter Test Cases + Generate Programs"
            overview_html += f'<div class="matrix-info">Operation: {round_type} | Programs: {len(eval_results)} | Test Cases: {len(tests)}</div>'
            
            # 生成矩阵表格
            overview_html += "<table>"
            
            # 表头：测试用例
            overview_html += "<tr><th>Program \\ Test</th>"
            for test_idx in range(len(tests)):
                overview_html += f"<th>T{test_idx + 1}</th>"
            overview_html += "</tr>"
            
            # 数据行：每个程序 vs 每个测试用例
            for prog_idx, eval_result in enumerate(eval_results):
                overview_html += f"<tr><td><strong>P{prog_idx + 1}</strong></td>"
                
                test_cases_pass_status = eval_result.get('test_cases_pass_status', [])
                for test_idx in range(len(tests)):
                    if test_idx < len(test_cases_pass_status):
                        passed = test_cases_pass_status[test_idx].get('pass', False)
                        css_class = "pass" if passed else "fail"
                        symbol = "✓" if passed else "✗"
                    else:
                        css_class = "na"
                        symbol = "N/A"
                    
                    overview_html += f'<td class="{css_class}">{symbol}</td>'
                
                overview_html += "</tr>"
            
            overview_html += "</table>"
            
            # 添加程序和测试用例的详细信息
            overview_html += "<details><summary>📋 View Test Cases</summary>"
            for test_idx, test in enumerate(tests):
                display_test = test if len(test) <= 80 else test[:77] + "..."
                overview_html += f'<div class="test-item"><strong>T{test_idx + 1}:</strong> {display_test}</div>'
            overview_html += "</details>"
            
            # 显示程序信息（如果有的话）
            programs = eval_results
            if programs:
                overview_html += "<details><summary>💻 View Programs Info</summary>"
                for prog_idx, prog_result in enumerate(programs):
                    pass_rate = prog_result.get('pass_rate', 0.0)
                    overview_html += f'<div class="program-item"><strong>P{prog_idx + 1}:</strong> Pass Rate: {pass_rate:.2%}</div>'
                overview_html += "</details>"
            
            overview_html += "</div>"
        
        overview_html += "</body></html>"
        
        # 保存总览文件
        overview_file = vis_dir / f"{qid}_dynamic_evolution.html"
        with open(overview_file, 'w', encoding='utf-8') as f:
            f.write(overview_html)
        print(f"📊 Dynamic evolution visualization saved to: {overview_file}")
        
        # 另外，为每一轮单独生成一个HTML文件
        for item in items_for_qid:
            round_num = item.get('round', 0)
            tests = item.get('synthesis_result', {}).get('tests') or []
            eval_results = item.get('gen_result', {}).get('eval_results') or []
            
            if not tests or not eval_results:
                continue
            
            # 单轮HTML
            round_html = f"<html><head><meta charset='UTF-8'><title>Round {round_num} Matrix - {qid}</title>"
            round_html += """
            <style>
                body { font-family: sans-serif; margin: 2em; }
                h1 { color: #333; }
                table { border-collapse: collapse; margin: 2em 0; width: 100%; }
                th, td { border: 1px solid #ccc; padding: 8px; text-align: center; }
                th { background-color: #f2f2f2; font-weight: bold; }
                .pass { background-color: #90EE90; }
                .fail { background-color: #FFB6C1; }
                .na { background-color: #e0e0e0; }
                .details { margin: 2em 0; }
                .test-item, .program-item { margin: 0.5em 0; font-family: monospace; font-size: 0.9em; }
            </style>
            """
            round_html += f"</head><body><h1>Round {round_num} Matrix for QID: {qid}</h1>"
            
            round_type = "Filter Programs + Generate Test Cases" if round_num % 2 == 0 else "Filter Test Cases + Generate Programs"
            round_html += f'<p><strong>Round Type:</strong> {round_type}</p>'
            round_html += f'<p><strong>Matrix Dimensions:</strong> {len(eval_results)} Programs × {len(tests)} Test Cases</p>'
            
            # 矩阵表格
            round_html += "<table>"
            round_html += "<tr><th>Program \\ Test Case</th>"
            for test_idx in range(len(tests)):
                round_html += f"<th>Test {test_idx + 1}</th>"
            round_html += "</tr>"
            
            for prog_idx, eval_result in enumerate(eval_results):
                round_html += f"<tr><td><strong>Program {prog_idx + 1}</strong></td>"
                
                test_cases_pass_status = eval_result.get('test_cases_pass_status', [])
                for test_idx in range(len(tests)):
                    if test_idx < len(test_cases_pass_status):
                        passed = test_cases_pass_status[test_idx].get('pass', False)
                        css_class = "pass" if passed else "fail"
                        symbol = "✓" if passed else "✗"
                    else:
                        css_class = "na"
                        symbol = "N/A"
                    
                    round_html += f'<td class="{css_class}">{symbol}</td>'
                
                round_html += "</tr>"
            
            round_html += "</table>"
            
            # 详细信息
            round_html += '<div class="details">'
            round_html += "<h2>Test Cases</h2>"
            for test_idx, test in enumerate(tests):
                round_html += f'<div class="test-item"><strong>Test {test_idx + 1}:</strong> {test}</div>'
            round_html += "</div>"
            
            round_html += "</body></html>"
            
            # 保存单轮文件
            round_file = vis_dir / f"{qid}_round{round_num}_matrix.html"
            with open(round_file, 'w', encoding='utf-8') as f:
                f.write(round_html)
        
        print(f"📊 Individual round matrices saved for QID: {qid}")

def generate_round_summary(vis_dir: Path):
    """Generate a summary HTML showing the evolution pattern across all rounds."""
    history_file = vis_dir / "visualization_history.jsonl"
    if not history_file.exists():
        return
    
    # Load all history data
    history_data = []
    with open(history_file, 'r') as f:
        for line in f:
            history_data.append(json.loads(line))
    
    # Group by round
    rounds_data = {}
    for item in history_data:
        round_num = item.get('round', 0)
        if round_num not in rounds_data:
            rounds_data[round_num] = []
        rounds_data[round_num].append(item)
    
    # Generate summary HTML
    summary_html = "<html><head><meta charset='UTF-8'><title>Multi-Round Adversarial Evolution Summary</title>"
    summary_html += """
    <style>
        body { font-family: sans-serif; margin: 2em; }
        h1 { color: #333; }
        .round-summary { margin: 2em 0; padding: 1em; border: 2px solid #ddd; border-radius: 8px; }
        .stats-table { border-collapse: collapse; margin: 1em 0; }
        .stats-table th, .stats-table td { border: 1px solid #ccc; padding: 8px; text-align: center; }
        .stats-table th { background-color: #f2f2f2; }
        .evolution-pattern { background-color: #f9f9f9; padding: 1em; border-radius: 5px; margin: 1em 0; }
        .pattern-odd { color: #d63384; }
        .pattern-even { color: #0d6efd; }
    </style>
    """
    summary_html += "</head><body><h1>🔄 Multi-Round Adversarial Evolution Summary</h1>"
    
    summary_html += '<div class="evolution-pattern">'
    summary_html += "<h2>🎯 Evolution Pattern</h2>"
    summary_html += "<p><strong>Round 0:</strong> Initial setup - filter both programs and test cases</p>"
    summary_html += '<p><strong class="pattern-odd">Odd Rounds (1,3,5...):</strong> Filter test cases + Generate new programs</p>'
    summary_html += '<p><strong class="pattern-even">Even Rounds (2,4,6...):</strong> Filter programs + Generate new test cases</p>'
    summary_html += "</div>"
    
    summary_html += "<h2>📊 Round-by-Round Statistics</h2>"
    summary_html += '<table class="stats-table">'
    summary_html += "<tr><th>Round</th><th>Operation Type</th><th>QIDs</th><th>Avg Programs</th><th>Avg Test Cases</th><th>Matrix Size</th></tr>"
    
    for round_num in sorted(rounds_data.keys()):
        round_data = rounds_data[round_num]
        
        # Calculate statistics
        total_qids = len(set(item.get('gen_result', {}).get('qid') for item in round_data))
        total_programs = sum(len(item.get('gen_result', {}).get('eval_results', [])) for item in round_data)
        total_tests = sum(len(item.get('synthesis_result', {}).get('tests') or []) for item in round_data)
        
        avg_programs = total_programs / len(round_data) if round_data else 0
        avg_tests = total_tests / len(round_data) if round_data else 0
        
        # Determine operation type
        if round_num == 0:
            operation = "Initial Setup"
        elif round_num % 2 == 1:
            operation = "Filter Test Cases + Gen Programs"
        else:
            operation = "Filter Programs + Gen Test Cases"
        
        matrix_size = f"{avg_programs:.1f} × {avg_tests:.1f}"
        
        summary_html += f"<tr><td>{round_num}</td><td>{operation}</td><td>{total_qids}</td>"
        summary_html += f"<td>{avg_programs:.1f}</td><td>{avg_tests:.1f}</td><td>{matrix_size}</td></tr>"
    
    summary_html += "</table>"
    
    # Add links to individual QID visualizations
    qids = set()
    for item in history_data:
        qid = item.get('gen_result', {}).get('qid')
        if qid:
            qids.add(qid)
    
    if qids:
        summary_html += "<h2>🔗 Individual QID Visualizations</h2>"
        summary_html += "<ul>"
        for qid in sorted(qids):
            summary_html += f'<li><a href="{qid}_dynamic_evolution.html">{qid} - Dynamic Evolution</a></li>'
        summary_html += "</ul>"
    
    summary_html += "</body></html>"
    
    # Save summary
    summary_file = vis_dir / "evolution_summary.html"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_html)
    
    print(f"📈 Evolution summary saved to: {summary_file}")

def main(
    file_path: str,
    output_dir: str = None,
    overwrite: bool = False,
    num_proc: int = 32,
    filter_mode: str = "auto",  # "auto" uses round-based alternating logic
):
    output_dir = Path(output_dir) if output_dir else Path(file_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    new_file_name = Path(file_path).stem.replace(LAST_STEP_NAME, FILE_NAME)
    output_file = output_dir / f"{new_file_name}.jsonl"
    stats_output_file = output_dir / f"{new_file_name}_stats.txt"
    
    # Extract round information from file path
    current_round = get_round_from_path(Path(file_path).name)
    
    # Determine filter mode based on round if set to auto
    if filter_mode == "auto":
        # Round 0: no filtering (preserve initial population completely)
        # Round 1+: alternating filtering to drive evolution
        # Odd rounds: filter test_cases (keep programs that survive, add new test cases)
        # Even rounds: filter programs (keep test cases that survive, add new programs)
        if current_round == 0:
            actual_filter_mode = "none"  # No filtering in round 0
        elif current_round % 2 == 1:  # Odd rounds
            actual_filter_mode = "test_case"
        else:  # Even rounds
            actual_filter_mode = "program"
        
        print(f"🔄 Round {current_round}: Auto-selected filter mode: {actual_filter_mode}")
    else:
        actual_filter_mode = filter_mode
        print(f"🔄 Round {current_round}: Manual filter mode: {actual_filter_mode}")
    
    if output_file.exists() and output_file.stat().st_size != 0 and not overwrite:
        print(f"Output file {output_file} already exists. Use --overwrite to overwrite.")
        with open(output_file, 'r') as f:
            data = [json.loads(line) for line in f.readlines()]
        print_statistics(data, output_file=stats_output_file)
        print(f"Returning cached data from {output_file}")
        return

    print(f"📥 Loading data from: {file_path}")
    
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f.readlines()]
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError("Unsupported file format. Please provide a .jsonl or .json file.")

    print(f"📥 Loaded {len(data)} problems")

    # Add round information to each item for visualization
    for item in data:
        item['round'] = current_round

    dataset = datasets.Dataset.from_list(data)
    
    # Apply filtering based on the determined mode
    if actual_filter_mode == "none":
        print(f"🚫 No filtering applied - preserving complete initial population")
    else:
        if actual_filter_mode in ["test_case", "both"]:
            print(f"🧹 Filtering test cases (removing ineffective ones)...")
            dataset = dataset.map(filter_test_cases, num_proc=num_proc, desc="Filtering test cases")
        
        if actual_filter_mode in ["program", "both"]:
            print(f"🧹 Filtering programs (removing those that fail all tests)...")
            dataset = dataset.map(filter_programs, num_proc=num_proc, desc="Filtering programs")

        # 🔧 FIX: Be more conservative about removing items to prevent disappearing problems
        # Only remove items that are truly empty, not just filtered down
        items_before_filter = len(dataset)
        dataset = dataset.filter(lambda item: len(item.get('synthesis_result', {}).get('tests') or []) > 0, num_proc=num_proc, desc="Removing items with no tests")
        dataset = dataset.filter(lambda item: len(item['gen_result'].get('eval_results', [])) > 0, num_proc=num_proc, desc="Removing items with no eval results")
        items_after_filter = len(dataset)
        
        if items_after_filter < items_before_filter:
            print(f"⚠️  WARNING: {items_before_filter - items_after_filter} problems completely removed due to lack of content.")
            print(f"    This may cause problems to 'disappear' from visualization.")
    
    print(f"📊 Filtering results: {len(data)} → {len(dataset)} items remaining")

    processed_dataset = list(dataset)

    # Note: Visualization is now handled by a separate script after all rounds are complete.
    # This script now focuses only on filtering.
    
    print(f"💾 Saving results to: {output_file}")
    with open(output_file, 'w') as f:
        for item in processed_dataset:
            item.pop('round', None)  # Remove temporary key before saving
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    if processed_dataset:
        print_statistics(processed_dataset, output_file=stats_output_file)
    else:
        with open(stats_output_file, 'w') as f:
            f.write(f"No items left after filtering in round {current_round}.\n")
            f.write(f"Filter mode used: {actual_filter_mode}\n")
            
    print(f"✅ Round {current_round} filtering complete. Results saved to {output_file}")
    
    # Don't return dataset to avoid printing it in terminal when called via fire.Fire()
    
if __name__ == "__main__":
    fire.Fire(main)

