import os
import fire
import json
from pathlib import Path
from subprocess import run
from typing import List, Optional, Dict
from datasets import Dataset

def pretty_name(name: str) -> str:
    """Convert a name to filesystem-friendly format."""
    return name.replace("/", "_").replace("-", "_").replace(" ", "_")

def parse_incomplete_json(json_str: str) -> dict:
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        json_str = json_str.replace("'", '"')
        return json.loads(json_str)
    
def parsing_item(item: dict) -> dict:
    ERROR_QUESTION = "Error in question generation"
    ERROR_TESTS = ["assert False"]
    
    gpt_response = item['synthesis_result'].get('gpt_response', {})
    
    if isinstance(gpt_response, dict):
        raw_text = gpt_response.get("message", {}).get("content", "")
    else:
        raw_text = str(gpt_response)
    
    try:
        obj = parse_incomplete_json(raw_text)
        question = obj.get("question", ERROR_QUESTION)
        tests = obj.get("tests", ERROR_TESTS)
    except Exception as e:
        print(f"Error parsing response: {e}")
        question = ERROR_QUESTION
        tests = ERROR_TESTS
    
    item['synthesis_result']['problem'] = question
    item['synthesis_result']['tests'] = tests
    return item

def run_python_file(script: str, args: List[str]):
    import sys
    cmd = [sys.executable, script] + args
    print(f"\n[Running] {' '.join(cmd)}")
    run(cmd, check=True)

def main(
    output_dir: str = "outputs/acecoder_rounds",
    model_name: str = "gpt-4.1-mini",
    use_vllm: bool = False,
    overwrite: bool = False,
    rounds: int = 1,
    max_tokens: int = 8448
):
    """
    Multi-round pipeline execution with consistent output folder structure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Unified directory structure
    dataset_dir = output_dir / "Magicoder_Evol_Instruct_110K" / model_name.replace("-", "_")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory structure:")
    print(f"- Base output dir: {output_dir}")
    print(f"- Dataset dir: {dataset_dir}")

    previous_result_file = None
    
    for i in range(rounds):
        print(f"\n================= 🔁 Round {i + 1} / {rounds} =================")

        # Unified file paths
        step1_output = dataset_dir / "step1_prompting_results.jsonl"
        step1_1_output = output_dir / f"step1.1_parsing_round{i}.jsonl"
        step2_1_output = output_dir / f"step2.1_gen_{model_name.replace('-', '_')}_seed42_round{i}.jsonl"
        step2_2_output = output_dir / f"step2.2_eval_{model_name.replace('-', '_')}_seed42_round{i}.jsonl"

        # === Step 1: prompting ===
        # Step1 should ONLY run in Round 0 to generate initial problems and test cases
        # All subsequent rounds work on these same problems, only generating new programs/test_cases
        if i == 0:
            # Round 0: Generate initial problems and test cases
            step1_generation_mode = "questions_and_tests"
            print(f"Round {i+1}: Generating initial problems and test cases")
            
            step1_args = [
                "--model_name", model_name,
                "--output_dir", str(output_dir),
                "--overwrite", str(overwrite),
                "--max_tokens", str(max_tokens),
                "--dataset_name", "ise-uiuc/Magicoder-Evol-Instruct-110K",
                "--generation_mode", step1_generation_mode,
            ]
            
            # Add API key if available
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                step1_args += ["--api_key", api_key]
            
            if previous_result_file:
                step1_args += ["--previous_result_file", str(previous_result_file)]
            
            run_python_file("step1_prompting.py", step1_args)
            
            # Parse the generated problems and test cases
            print(f"Round {i+1}: Parsing initial problems and test cases")
            run_python_file("step1.1_parsing.py", [
                "--file_path", str(step1_output),
                "--output_dir", str(output_dir),
                "--overwrite", str(overwrite),
                "--parsing_mode", step1_generation_mode,
            ])
        else:
            # Round 1+: Skip step1 entirely, use existing problems from Round 0
            print(f"Round {i+1}: Using existing problems from Round 0 (no new problem generation)")
            # Ensure we have the Round 0 parsing results available for step2
            round0_parsing = output_dir / "step1.1_parsing_round0.jsonl"
            if not round0_parsing.exists():
                raise FileNotFoundError(f"Round 0 parsing results not found: {round0_parsing}. Cannot proceed with Round {i+1}.")

        # Handle parsing output for Round 0 or verify Round 0 parsing exists for subsequent rounds
        if i == 0:
            # Verify Round 0 parsing was successful
            default_parsing_output = output_dir / "step1.1_parsing.jsonl"
            if default_parsing_output.exists():
                default_parsing_output.rename(step1_1_output)
            else:
                raise FileNotFoundError(f"Round 0 parsing output not found: {default_parsing_output}")
        else:
            # For Round 1+, use Round 0 parsing as the source of problems
            step1_1_output = output_dir / "step1.1_parsing_round0.jsonl"
            if not step1_1_output.exists():
                raise FileNotFoundError(f"Round 0 parsing results required for Round {i+1}: {step1_1_output}")

        # === Step 2.1: generation ===
        if use_vllm:
            run_python_file("step2.1_vllm_gen.py", [
                str(step1_1_output),
                "--model_name_or_path", model_name,
                "--output_dir", str(output_dir),
                "--overwrite", str(overwrite),
            ])
        else:
            # Determine generation mode based on round number
            # 🔧 FIX: Correct adversarial evolution logic
            # Round 0: Initialize problems + programs + test cases (complete setup)
            # Odd rounds (1,3,5...): Clean programs → Generate new programs → Eval
            # Even rounds (2,4,6...): Clean test cases → Generate new test cases → Eval
            if i == 0:
                generation_mode = "programs"  # Generate initial programs to go with problems+tests
                print(f"\n🚀 Round {i}: Complete initialization (problems + programs + test cases)")
            elif i % 2 == 1:
                generation_mode = "programs"  # Odd rounds: generate new programs
                print(f"\n🔄 Round {i}: Clean & generate new PROGRAMS (after filtering)")
            else:
                generation_mode = "test_cases"  # Even rounds: generate new test cases
                print(f"\n🔄 Round {i}: Clean & generate new TEST CASES (after filtering)")
            
            # 🔧 FIX: Use the correct input file for each round
            # Round 0: Use step1_1_output (initial parsing)
            # Round 1+: Use previous_result_file (previous round's filtered results)
            if i == 0:
                input_file = str(step1_1_output)
            else:
                input_file = str(previous_result_file) if previous_result_file else str(step1_1_output)
            
            step2_args = [
                input_file,
                "--model", model_name,
                "--output_dir", str(output_dir),
                "--overwrite", str(overwrite),
                "--generation_mode", generation_mode,
            ]
            
            # Add API key if available
            if api_key:
                step2_args += ["--api_key", api_key]
            
            run_python_file("step2.1_openai_gen.py", step2_args)

        # Handle generation output
        default_step2_1_output = output_dir / f"step2.1_gen_{model_name.replace('-', '_')}_seed42.jsonl"
        if default_step2_1_output.exists():
            default_step2_1_output.rename(step2_1_output)
        else:
            raise FileNotFoundError(f"Generation output not found: {default_step2_1_output}")

        # # === Step 2.2: evaluation ===
        # if step2_1_output.exists():
        #     # Debug: print first 3 lines of input
        #     print("\nDebug - step2.1 output sample:")
        #     with open(step2_1_output, 'r') as f:
        #         for i, line in enumerate(f):
        #             if i >= 3: break
        #             try:
        #                 data = json.loads(line)
        #                 print(f"Line {i+1}: {json.dumps(data, indent=2, ensure_ascii=False)[:200]}...")
        #             except json.JSONDecodeError:
        #                 print(f"Line {i+1}: (Invalid JSON) {line.strip()[:200]}...")

        #     filter_mode = "program" if i % 2 == 1 else "test_case"
        #     print(f"\nRound {i+1}: Using {filter_mode.upper()} filtering mode")

        #     run_python_file("step2.2_eval.py", [
        #         str(step2_1_output),
        #         "--output_dir", str(output_dir),
        #         "--overwrite", str(overwrite).lower(),
        #         "--max_samples", "2",
        #     ])

        #     # Verify evaluation output
        #     eval_output = output_dir / f"step2.2_eval_{model_name.replace('-', '_')}_seed42_round{i}.jsonl"
        #     if not eval_output.exists():
        #         raise FileNotFoundError(f"Evaluation output not found: {eval_output}")

        #     # === Step 3: filtering ===
        #     run_python_file("step_3_filter_tests.py", [
        #         str(eval_output),
        #         "--output_dir", str(output_dir),
        #         "--overwrite", str(overwrite).lower(),
        #         "--filter_mode", filter_mode,
        #         "--num_proc", "1"
        #     ])

        #     previous_result_file = eval_output
        # else:
        #     raise FileNotFoundError(f"Missing required input file: {step2_1_output}")


        # === Step 2.2: evaluation ===
        if step2_1_output.exists():
            # Debug: Show basic info about generation output (without dumping JSON)
            print(f"\n📊 Round {i+1} Generation Summary:")
            with open(step2_1_output, 'r') as f:
                lines = [line for line in f if line.strip()]
                print(f"   Generated data for {len(lines)} problems")
                if lines:
                    try:
                        sample = json.loads(lines[0])
                        qid = sample.get('gen_result', {}).get('qid', 'unknown')[:8]
                        outputs = len(sample.get('gen_result', {}).get('outputs', []))
                        print(f"   Sample problem {qid}...: {outputs} outputs")
                    except:
                        print(f"   Sample: (parsing error)")

            # Determine what to evaluate based on generation mode
            if generation_mode == "test_cases":
                # For test case generation rounds, merge new test cases with existing programs
                print(f"\n🔄 Round {i+1}: Merging new test cases with existing programs...")
                
                if not previous_result_file or not previous_result_file.exists():
                    raise FileNotFoundError(f"Previous round result file not found: {previous_result_file}")
                
                # Create merged file
                merged_output = output_dir / f"step2.1_merged_gpt_4.1_mini_seed42_round{i}.jsonl"
                
                run_python_file("merge_test_cases.py", [
                    str(previous_result_file),  # programs from previous round
                    str(step2_1_output),        # test cases from current round
                    str(merged_output)          # merged output
                ])
                
                eval_input_file = merged_output
            else:
                # For program generation rounds, evaluate the newly generated programs
                eval_input_file = step2_1_output
                
            print(f"\n🔍 Round {i+1}: Evaluating programs against test cases...")
            print(f"📁 Using evaluation input: {eval_input_file}")
            
            run_python_file("step2.2_eval.py", [
                str(eval_input_file),
                "--output_dir", str(output_dir),
                "--overwrite", str(overwrite).lower(),
                "--max_samples", "100",
                "--current_round", str(i),
            ])

            # Verify evaluation output
            eval_output = output_dir / f"step2.2_eval_{model_name.replace('-', '_')}_seed42_round{i}.jsonl"
            if not eval_output.exists():
                raise FileNotFoundError(f"Evaluation output not found: {eval_output}")

            # === Step 3: filtering ===
            # Use auto mode for alternating filter logic based on round number
            print(f"\n🧹 Round {i+1}: Applying intelligent filtering...")
            
            run_python_file("step_3_filter_tests.py", [
                str(eval_output),
                "--output_dir", str(output_dir),
                "--overwrite", str(overwrite).lower(),
                "--filter_mode", "auto",  # Let the filter function decide based on round
                "--num_proc", "1"
            ])

            # Update the filtered results as the next round's input
            filtered_output = output_dir / f"step_3_filter_tests_{model_name.replace('-', '_')}_seed42_round{i}.jsonl"
            if filtered_output.exists():
                previous_result_file = filtered_output
            else:
                previous_result_file = eval_output
                
        else:
            raise FileNotFoundError(f"Missing required input file: {step2_1_output}")

    # After all rounds are complete, generate the final visualization
    vis_dir = output_dir / "visualizations"
    history_file = vis_dir / "visualization_history.jsonl"
    
    if history_file.exists():
        print(f"\n================= 🎨 Generating Final Visualizations =================")
        
        # Generate traditional HTML visualizations
        run_python_file("generate_visuals.py", [
            "--history_file", str(history_file),
            "--output_dir", str(output_dir),
        ])
        print(f"📊 Traditional HTML visualizations are ready in: {vis_dir}")
        
        # Launch modern Gradio interface
        print(f"\n🚀 Launching Advanced Gradio Visualizer...")
        print(f"🌐 Starting interactive web interface...")
        print(f"💡 You can also run manually: python launch_visualizer.py")
        
        print(f"✅ Pipeline completed! Visualization data is ready.")
        print(f"📊 Visualization files generated in: {history_file.parent}")
        print(f"💡 The visualization will be automatically updated in the integrated interface.")
            
    else:
        print(f"\n⚠️ History file not found, skipping final visualization.")

    print(f"\n🎉 All {rounds} rounds completed successfully!")
    print(f"📊 Check all outputs in: {output_dir}")
    print(f"🎨 Access interactive visualizer at: http://localhost:7860")
    print("-" * 60)

if __name__ == "__main__":
    fire.Fire(main)