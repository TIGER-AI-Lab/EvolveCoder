#!/usr/bin/env python3
"""
🚀 Integrated Gradio Application for Adversarial Generation

This application provides a complete interface for:
1. Configuring and running the adversarial generation pipeline
2. Real-time monitoring of execution progress
3. Interactive visualization of results
4. Live feedback and logging display
"""

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
import numpy as np
from pathlib import Path
import subprocess
import threading
import time
import queue
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import os
import sys

class IntegratedAdvancedSystem:
    """Integrated system for adversarial generation with real-time monitoring."""
    
    def __init__(self):
        self.output_dir = Path("outputs/acecoder_rounds")
        self.process = None
        self.log_queue = queue.Queue()
        self.is_running = False
        self.current_round = 0
        self.total_rounds = 1
        self.progress = 0.0
        self.data_cache = {}
        self.show_backend_logs = True  # Default to showing backend logs
        self.results_updated = False  # Flag to indicate new results available
    
    def set_backend_logging(self, enabled: bool):
        """Enable or disable backend terminal logging."""
        self.show_backend_logs = enabled
        if enabled:
            print("🖥️ Backend logging enabled - logs will show in terminal")
        else:
            print("🔇 Backend logging disabled - logs only in frontend")
        
    def run_pipeline(self, rounds: int, model_name: str, max_tokens: int, 
                    output_dir: str, overwrite: bool, progress_callback=None):
        """Run the adversarial generation pipeline with real-time feedback."""
        
        self.is_running = True
        self.current_round = 0
        self.total_rounds = rounds
        self.progress = 0.0
        
        # Clear previous logs
        while not self.log_queue.empty():
            self.log_queue.get()
        
        def log_output(message, show_in_terminal=True):
            """Log to both frontend queue and backend terminal."""
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_msg = f"[{timestamp}] {message}"
            
            # Always add to frontend queue
            self.log_queue.put(formatted_msg)
            
            # Also show in backend terminal if enabled
            if show_in_terminal and self.show_backend_logs:
                print(formatted_msg, flush=True)
            
            if progress_callback:
                progress_callback()
        
        def run_command():
            try:
                log_output("🚀 Starting adversarial generation pipeline...")
                log_output(f"📊 Configuration: {rounds} rounds, model: {model_name}")
                
                # Build command
                cmd = [
                    sys.executable, "main.py",
                    "--output_dir", output_dir,
                    "--model_name", model_name,
                    "--rounds", str(rounds),
                    "--max_tokens", str(max_tokens),
                ]
                if overwrite:
                    cmd.append("--overwrite")
                
                log_output(f"🔧 Command: {' '.join(cmd)}")
                
                # Set up environment with API keys
                env = os.environ.copy()
                if not env.get('OPENAI_API_KEY'):
                    log_output("⚠️ Warning: OPENAI_API_KEY not found in environment")
                
                # Start process with inherited environment
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,
                    env=env
                )
                
                # Monitor output
                for line in iter(self.process.stdout.readline, ''):
                    if line.strip():
                        # Parse round information
                        round_match = re.search(r'Round (\d+) / (\d+)', line)
                        if round_match:
                            self.current_round = int(round_match.group(1))
                            self.total_rounds = int(round_match.group(2))
                            self.progress = (self.current_round / self.total_rounds) * 100
                        
                        # Parse step information
                        if "Step1" in line:
                            log_output("📝 Step 1: Generating prompts...")
                        elif "Step2.1" in line:
                            log_output("🤖 Step 2.1: Generating content...")
                        elif "Step2.2" in line:
                            log_output("📊 Step 2.2: Evaluating results...")
                        elif "Step 3" in line:
                            log_output("🧹 Step 3: Filtering and processing...")
                        elif "Gradio" in line:
                            log_output("🎨 Launching visualization interface...")
                        
                        log_output(line.strip())
                
                self.process.wait()
                
                if self.process.returncode == 0:
                    log_output("✅ Pipeline completed successfully!")
                    self.progress = 100.0
                    log_output("📊 Loading visualization results...")
                    self.load_results()
                    self.results_updated = True  # Mark that new results are available
                    log_output("🎨 Visualization data ready! Check the Visualization tab for updated results.")
                else:
                    log_output(f"❌ Pipeline failed with return code: {self.process.returncode}")
                
            except Exception as e:
                log_output(f"❌ Error running pipeline: {str(e)}")
            finally:
                self.is_running = False
                self.process = None
        
        # Run in separate thread
        thread = threading.Thread(target=run_command)
        thread.daemon = True
        thread.start()
        
        return thread
    
    def stop_pipeline(self):
        """Stop the running pipeline."""
        if self.process:
            self.process.terminate()
            self.process = None
            self.is_running = False
            self.log_queue.put("[SYSTEM] Pipeline stopped by user")
    
    def get_logs(self) -> str:
        """Get accumulated logs for display."""
        new_logs = []
        while not self.log_queue.empty():
            new_logs.append(self.log_queue.get())
        return "\n".join(new_logs) if new_logs else ""
    
    def get_all_logs(self) -> str:
        """Get all logs accumulated so far."""
        # Store logs in a persistent list
        if not hasattr(self, '_all_logs'):
            self._all_logs = []
        
        # Add new logs to the persistent list
        while not self.log_queue.empty():
            self._all_logs.append(self.log_queue.get())
        
        # Return recent logs (last 50 lines to prevent overflow)
        recent_logs = self._all_logs[-50:] if len(self._all_logs) > 50 else self._all_logs
        return "\n".join(recent_logs)
    
    def get_status(self) -> Dict:
        """Get current execution status."""
        return {
            "is_running": self.is_running,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "progress": self.progress,
            "status_text": f"Round {self.current_round}/{self.total_rounds}" if self.is_running else "Ready"
        }
    
    def load_results(self):
        """Load visualization results after pipeline completion."""
        history_file = self.output_dir / "visualizations" / "visualization_history.jsonl"
        
        if not history_file.exists():
            self.data_cache = {"error": "No results found"}
            return
        
        try:
            history_data = []
            with open(history_file, 'r') as f:
                for line in f:
                    history_data.append(json.loads(line))
            
            # Group by QID
            self.data_cache = {}
            for item in history_data:
                qid = item.get('gen_result', {}).get('qid')
                if qid:
                    if qid not in self.data_cache:
                        self.data_cache[qid] = []
                    self.data_cache[qid].append(item)
            
            # Sort by round number
            for qid in self.data_cache:
                self.data_cache[qid].sort(key=lambda x: x.get('round', 0))
                
        except Exception as e:
            self.data_cache = {"error": f"Error loading results: {str(e)}"}
    
    def get_qid_list(self) -> List[str]:
        """Get list of available QIDs, prioritizing those with multiple rounds."""
        if "error" in self.data_cache:
            return []
        
        # Separate QIDs by number of rounds (prioritize multi-round QIDs)
        multi_round_qids = []
        single_round_qids = []
        
        for qid, data in self.data_cache.items():
            rounds = len(set(item.get('round', 0) for item in data))
            if rounds > 1:
                multi_round_qids.append(qid)
            else:
                single_round_qids.append(qid)
        
        # Return multi-round QIDs first, then single-round ones
        return sorted(multi_round_qids) + sorted(single_round_qids)
    
    def create_program_test_matrix(self, qid: str, round_num: int) -> go.Figure:
        """Create a program-test matrix for a specific round: rows=programs, columns=tests."""
        if not qid or qid not in self.data_cache:
            fig = go.Figure()
            fig.add_annotation(text="No data available - run pipeline first", 
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font_size=16)
            return fig

        rounds_data = self.data_cache[qid]
        if not rounds_data:
            fig = go.Figure()
            fig.add_annotation(text="No rounds data available", 
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font_size=16)
            return fig

        # Find data for the specific round
        round_data = None
        for item in rounds_data:
            if item.get('round', 0) == round_num:
                round_data = item
                break
        
        if not round_data:
            fig = go.Figure()
            fig.add_annotation(text=f"No data available for Round {round_num}", 
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font_size=16)
            return fig

        # Get tests and evaluation results
        tests = round_data.get('synthesis_result', {}).get('tests', [])
        eval_results = round_data.get('gen_result', {}).get('eval_results', [])
        programs = round_data.get('programs', [round_data.get('program', '')])
        
        if not tests:
            fig = go.Figure()
            fig.add_annotation(text="No test cases found", 
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font_size=16)
            return fig
        
        if not eval_results:
            fig = go.Figure()
            fig.add_annotation(text="No evaluation results found", 
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font_size=16)
            return fig

        # Build matrix: rows = programs, columns = tests
        matrix = []
        program_labels = []
        test_labels = [f"t{i+1}" for i in range(len(tests))]
        hover_text = []
        
        # Create matrix for each program
        for prog_idx, eval_result in enumerate(eval_results):
            program_labels.append(f"p{prog_idx + 1}")
            row = []
            hover_row = []
            test_cases_pass_status = eval_result.get('test_cases_pass_status', [])
            
            for test_idx in range(len(tests)):
                if test_idx < len(test_cases_pass_status):
                    test_result = test_cases_pass_status[test_idx]
                    if isinstance(test_result, dict):
                        passed = 1 if test_result.get('pass', False) else 0
                        error_msg = test_result.get('error_message', '')
                    else:
                        passed = 1 if test_result else 0
                        error_msg = ''
                    
                    row.append(passed)
                    status_text = "PASS" if passed else "FAIL"
                    error_display = (error_msg[:100] + "...") if error_msg else "No error message"
                    hover_row.append(f"Program {prog_idx+1}<br>Test {test_idx+1}: {status_text}<br>{error_display}")
                else:
                    row.append(-1)  # Missing test result
                    hover_row.append(f"Program {prog_idx+1}<br>Test {test_idx+1}: N/A")
            
            matrix.append(row)
            hover_text.append(hover_row)
        
        # Handle case where no programs were evaluated
        if not matrix:
            program_labels.append("p1")
            row = [-1] * len(tests)
            hover_row = [f"Program 1<br>Test {i+1}: N/A" for i in range(len(tests))]
            matrix.append(row)
            hover_text.append(hover_row)
        
        # Create custom colorscale with clear color mapping
        colorscale = [
            [0.0, '#FF0000'],    # Red for FAIL (0)
            [0.5, '#CCCCCC'],    # Gray for N/A (-1)  
            [1.0, '#00FF00']     # Green for PASS (1)
        ]
        
        # Convert matrix values for display with direct mapping
        display_matrix = []
        for row in matrix:
            display_row = []
            for val in row:
                if val == -1:
                    display_row.append(-1)   # Gray (N/A)
                elif val == 0:
                    display_row.append(0)    # Red (FAIL)
                else:
                    display_row.append(1)    # Green (PASS)
            display_matrix.append(display_row)
        
        fig = go.Figure(data=go.Heatmap(
            z=display_matrix,
            x=test_labels,
            y=program_labels,
            hovertext=hover_text,
            hovertemplate="%{hovertext}<extra></extra>",
            colorscale=colorscale,
            zmid=0,  # Center the colorscale at 0
            showscale=False,
            xgap=2,
            ygap=2
        ))
        
        fig.update_layout(
            title=f"Program-Test Matrix - Round {round_num} (QID: {qid[:8]}...)",
            xaxis_title="Test Cases",
            yaxis_title="Programs", 
            width=800,
            height=400,
            font=dict(size=12),
            plot_bgcolor='white'
        )
        
        fig.update_xaxes(side="top")
        fig.update_yaxes(autorange="reversed")
        
        return fig
        
    def create_all_rounds_matrices(self, qid: str) -> go.Figure:
        """Create subplots showing program-test matrices for all rounds."""
        if not qid or qid not in self.data_cache:
            fig = go.Figure()
            fig.add_annotation(text="No data available - run pipeline first", 
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font_size=16)
            return fig

        rounds_data = self.data_cache[qid]
        if not rounds_data:
            fig = go.Figure()
            fig.add_annotation(text="No rounds data available", 
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font_size=16)
            return fig

        # Get all available rounds
        available_rounds = sorted(set(item.get('round', 0) for item in rounds_data))
        num_rounds = len(available_rounds)
        
        if num_rounds == 0:
            fig = go.Figure()
            fig.add_annotation(text="No valid rounds found", 
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font_size=16)
            return fig
        
        # Calculate subplot layout
        cols = min(3, num_rounds)  # Max 3 columns
        rows = (num_rounds + cols - 1) // cols  # Ceiling division
        
        fig = make_subplots(
            rows=rows, 
            cols=cols,
            subplot_titles=[f"Round {r}" for r in available_rounds],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Get maximum test count across all rounds for consistent grid layout
        max_test_count = 0
        for round_data in rounds_data:
            tests = round_data.get('synthesis_result', {}).get('tests', [])
            max_test_count = max(max_test_count, len(tests))
        
        # Use max test count for consistent grid, but each round will show actual tests
        max_test_labels = [f"t{i+1}" for i in range(max_test_count)]
        
        # Create matrix for each round
        for idx, round_num in enumerate(available_rounds):
            row_pos = (idx // cols) + 1
            col_pos = (idx % cols) + 1
            
            # Find round data
            round_data = None
            for item in rounds_data:
                if item.get('round', 0) == round_num:
                    round_data = item
                    break
            
            if round_data:
                eval_results = round_data.get('gen_result', {}).get('eval_results', [])
                programs = round_data.get('programs', [round_data.get('program', '')])
                current_tests = round_data.get('synthesis_result', {}).get('tests', [])
                
                # Build matrix for this round using actual test count
                matrix = []
                program_labels = []
                test_labels = [f"t{i+1}" for i in range(len(current_tests))]
                
                for prog_idx, eval_result in enumerate(eval_results):
                    program_labels.append(f"p{prog_idx + 1}")
                    row = []
                    test_cases_pass_status = eval_result.get('test_cases_pass_status', [])
                    
                    # Use actual test count for this round
                    for test_idx in range(len(current_tests)):
                        if test_idx < len(test_cases_pass_status):
                            test_result = test_cases_pass_status[test_idx]
                            if isinstance(test_result, dict):
                                passed = 1 if test_result.get('pass', False) else 0
                            else:
                                passed = 1 if test_result else 0
                            row.append(passed)
                        else:
                            row.append(-1)  # Missing data
                    
                    matrix.append(row)
                
                if not matrix:  # No evaluation results
                    program_labels = ["p1"]
                    matrix = [[-1] * len(current_tests)]
                
                # Convert for display with clear color mapping
                display_matrix = []
                for row in matrix:
                    display_row = []
                    for val in row:
                        if val == -1:
                            display_row.append(-1)   # Gray (N/A)
                        elif val == 0:
                            display_row.append(0)    # Red (FAIL)
                        else:
                            display_row.append(1)    # Green (PASS)
                    display_matrix.append(display_row)
                
                fig.add_trace(
                    go.Heatmap(
                        z=display_matrix,
                        x=test_labels,
                        y=program_labels,
                        colorscale=[
                            [0.0, '#FF0000'],    # Red for FAIL (0)
                            [0.5, '#CCCCCC'],    # Gray for N/A (-1)  
                            [1.0, '#00FF00']     # Green for PASS (1)
                        ],
                        zmid=0,  # Center the colorscale at 0
                        showscale=False,
                        xgap=1,
                        ygap=1
                    ),
                    row=row_pos, col=col_pos
                )
        
        fig.update_layout(
            title=f"All Rounds Program-Test Matrices (QID: {qid[:8]}...)",
            height=300 * rows,
            width=1000,
            font=dict(size=10)
        )
        
        return fig

    def create_matrix_heatmap(self, qid: str, round_num: int) -> go.Figure:
        """Create matrix visualization (same as before)."""
        if not qid or qid not in self.data_cache:
            fig = go.Figure()
            fig.add_annotation(text="No data available - run pipeline first", 
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font_size=16)
            return fig
        
        # Find the round data
        round_data = None
        for item in self.data_cache[qid]:
            if item.get('round', 0) == round_num:
                round_data = item
                break
        
        if not round_data:
            fig = go.Figure()
            fig.add_annotation(text=f"No data for round {round_num}", 
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font_size=16)
            return fig
        
        tests = round_data.get('synthesis_result', {}).get('tests') or []
        eval_results = round_data.get('gen_result', {}).get('eval_results') or []
        
        if not tests or not eval_results:
            fig = go.Figure()
            fig.add_annotation(text="No evaluation data available", 
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font_size=16)
            return fig
        
        # Build matrix
        matrix = []
        program_labels = []
        test_labels = []
        hover_text = []
        
        for prog_idx, eval_result in enumerate(eval_results):
            program_labels.append(f"Program {prog_idx + 1}")
            row = []
            hover_row = []
            test_cases_pass_status = eval_result.get('test_cases_pass_status', [])
            
            for test_idx, test in enumerate(tests):
                if prog_idx == 0:  # Only add test labels once
                    test_labels.append(f"Test {test_idx + 1}")
                
                if test_idx < len(test_cases_pass_status):
                    test_result = test_cases_pass_status[test_idx]
                    if isinstance(test_result, dict):
                        passed = test_result.get('pass', False)
                    else:
                        passed = bool(test_result)  # Handle boolean values
                    
                    row.append(2 if passed else 0)  # 🔧 FIX: Use 2 for pass, 0 for fail, 1 for N/A
                    status = "✅ PASS" if passed else "❌ FAIL"
                    test_preview = test[:60] + "..." if len(test) > 60 else test
                    hover_row.append(f"Program {prog_idx + 1}<br>Test {test_idx + 1}<br>Status: {status}<br><br>Test: {test_preview}")
                else:
                    row.append(1)  # N/A
                    hover_row.append(f"Program {prog_idx + 1}<br>Test {test_idx + 1}<br>Status: N/A")
            
            matrix.append(row)
            hover_text.append(hover_row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=test_labels,
            y=program_labels,
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_text,
            colorscale=[
                [0, '#FF6B6B'],    # 0 = Fail - Red
                [0.5, '#FFE066'],  # 1 = N/A - Yellow  
                [1, '#4ECDC4']     # 2 = Pass - Green
            ],
            colorbar=dict(
                title="Result",
                tickvals=[0, 1, 2],
                ticktext=["Fail", "N/A", "Pass"]
            )
        ))
        
        round_type = "Generate Programs" if round_num % 2 == 1 else "Generate Tests"
        
        fig.update_layout(
            title=f"Round {round_num} Matrix - {round_type}<br>QID: {qid[:16]}...",
            xaxis_title="Test Cases",
            yaxis_title="Programs",
            font=dict(size=12),
            width=800,
            height=500
        )
        
        return fig

def create_integrated_interface():
    """Create the integrated Gradio interface."""
    
    system = IntegratedAdvancedSystem()
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
    .gr-button-primary {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
    }
    .gr-button-secondary {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%) !important;
        border: none !important;
        color: white !important;
    }
    .log-container {
        background-color: #1e1e1e !important;
        color: #00ff00 !important;
        font-family: 'Courier New', monospace !important;
        font-size: 12px !important;
        padding: 10px !important;
        border-radius: 5px !important;
        max-height: 400px !important;
        overflow-y: auto !important;
    }
    .progress-bar {
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%) !important;
        border-radius: 10px !important;
    }
    """
    
    with gr.Blocks(
        title="🚀 Integrated Adversarial Generation System", 
        theme=gr.themes.Soft(primary_hue="blue"),
        css=custom_css
    ) as app:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1>🚀 Integrated Adversarial Generation System</h1>
            <p>Configure, run, and visualize multi-round adversarial generation in real-time</p>
        </div>
        """)
        
        with gr.Tab("🎯 Pipeline Control"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<h3>⚙️ Configuration</h3>")
                    
                    rounds_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=1,
                        label="Number of Rounds",
                        info="How many adversarial rounds to run"
                    )
                    
                    model_dropdown = gr.Dropdown(
                        choices=[
                            "gpt-4.1-mini",
                            "gpt-4",
                            "gpt-3.5-turbo",
                            "gpt-4-turbo"
                        ],
                        value="gpt-4.1-mini",
                        label="Model Name",
                        info="OpenAI model to use for generation"
                    )
                    
                    max_tokens_slider = gr.Slider(
                        minimum=1000,
                        maximum=10000,
                        step=500,
                        value=4000,
                        label="Max Tokens",
                        info="Maximum tokens per API call"
                    )
                    
                    output_dir_text = gr.Textbox(
                        value="outputs/acecoder_rounds",
                        label="Output Directory",
                        info="Where to save results"
                    )
                    
                    overwrite_checkbox = gr.Checkbox(
                        value=True,
                        label="Overwrite Existing Files",
                        info="Whether to overwrite existing outputs"
                    )
                    
                    gr.HTML("<h4>🔑 API Configuration</h4>")
                    
                    api_key_text = gr.Textbox(
                        value=os.getenv('OPENAI_API_KEY', ''),
                        label="OpenAI API Key",
                        placeholder="sk-proj-...",
                        type="password",
                        info="Your OpenAI API key (will be masked)"
                    )
                    
                    backend_logging_checkbox = gr.Checkbox(
                        value=True,
                        label="🖥️ Show Backend Logs",
                        info="Display logs in terminal (backend) in addition to frontend"
                    )
                    
                    with gr.Row():
                        start_btn = gr.Button("🚀 Start Pipeline", variant="primary", size="lg")
                        stop_btn = gr.Button("🛑 Stop Pipeline", variant="secondary", size="lg")
                
                with gr.Column(scale=2):
                    gr.HTML("<h3>📊 Real-time Status</h3>")
                    
                    status_text = gr.Textbox(
                        value="Ready to start",
                        label="Current Status",
                        interactive=False
                    )
                    
                    progress_bar = gr.Progress()
                    
                    gr.HTML("<h3>📋 Live Logs</h3>")
                    log_display = gr.Textbox(
                        value="",
                        label="Execution Logs",
                        lines=15,
                        max_lines=20,
                        interactive=False,
                        elem_classes=["log-container"]
                    )
        
        with gr.Tab("📊 Real-time Visualization"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<h3>🎮 Visualization Controls</h3>")
                    
                    qid_dropdown = gr.Dropdown(
                        choices=[],
                        label="Select QID",
                        info="Choose a Question ID to visualize"
                    )
                    
                    viz_type_radio = gr.Radio(
                        choices=["All Rounds (Grid)", "Single Round", "Legacy Multi-Round"],
                        value="All Rounds (Grid)",
                        label="Visualization Type",
                        info="Choose visualization type"
                    )
                    
                    round_viz_slider = gr.Slider(
                        minimum=0,
                        maximum=10,
                        step=1,
                        value=0,
                        label="Round Number",
                        info="Select which round to visualize (for Single Round Matrix)",
                        visible=False
                    )
                    
                    refresh_viz_btn = gr.Button("🔄 Refresh Visualization", variant="primary")
                    load_results_btn = gr.Button("📊 Load Results", variant="secondary")
            
            with gr.Row():
                matrix_plot = gr.Plot(label="Program-Test Results Matrix")
        
        with gr.Tab("📈 Results Summary"):
            gr.HTML("<h3>📋 Pipeline Summary</h3>")
            summary_text = gr.Markdown("Run the pipeline to see results summary")
        
        # State variables
        log_state = gr.State("")
        
        # Event handlers
        def start_pipeline(rounds, model_name, max_tokens, output_dir, overwrite, api_key, backend_logging):
            if system.is_running:
                return "⚠️ Pipeline already running!", ""
            
            # Set API key in environment if provided
            if api_key.strip():
                os.environ['OPENAI_API_KEY'] = api_key.strip()
            elif not os.getenv('OPENAI_API_KEY'):
                return "❌ Please provide OpenAI API Key!", ""
            
            # Configure backend logging
            system.set_backend_logging(backend_logging)
            
            # Start the pipeline
            system.run_pipeline(rounds, model_name, max_tokens, output_dir, overwrite)
            
            return "🚀 Pipeline started...", ""
        
        def stop_pipeline():
            system.stop_pipeline()
            return "🛑 Pipeline stopped", ""
        
        def update_status():
            status = system.get_status()
            all_logs = system.get_all_logs()
            
            # Check if pipeline just completed and results are available
            if not status['is_running'] and system.results_updated:
                status_msg = f"✅ Pipeline completed! New visualization data available - Check Visualization tab"
            else:
                status_msg = f"{'🔄 Running' if status['is_running'] else '✅ Ready'} - {status['status_text']}"
            
            progress = status['progress'] / 100.0 if status['progress'] > 0 else 0
            
            return status_msg, all_logs, progress
        
        def refresh_visualization():
            # Force reload results regardless of flag
            system.load_results()
            qid_choices = system.get_qid_list()
            
            # Return updated components - default to all rounds grid
            updated_dropdown = gr.Dropdown(choices=qid_choices, value=qid_choices[0] if qid_choices else None)
            
            if qid_choices:
                try:
                    updated_plot = system.create_all_rounds_matrices(qid_choices[0])
                except Exception as e:
                    print(f"❌ Error in refresh_visualization: {e}")
                    # Create error figure
                    updated_plot = go.Figure()
                    updated_plot.add_annotation(text=f"Error loading visualization: {str(e)}", 
                                              xref="paper", yref="paper", x=0.5, y=0.5, 
                                              showarrow=False, font_size=16)
            else:
                updated_plot = None
                
            updated_summary = get_summary()
            
            return updated_dropdown, updated_plot, updated_summary
        
        def update_visualization(qid, viz_type, round_num):
            if not qid:
                return None
            
            try:
                if viz_type == "All Rounds (Grid)":
                    return system.create_all_rounds_matrices(qid)
                elif viz_type == "Single Round":
                    return system.create_program_test_matrix(qid, round_num)
                else:  # Legacy Multi-Round
                    return system.create_matrix_heatmap(qid, round_num)
            except Exception as e:
                print(f"❌ Error in update_visualization: {e}")
                # Return empty figure on error
                fig = go.Figure()
                fig.add_annotation(text=f"Error: {str(e)}", 
                                 xref="paper", yref="paper", x=0.5, y=0.5, 
                                 showarrow=False, font_size=16)
                return fig
        
        def toggle_round_slider(viz_type):
            """Show/hide round slider based on visualization type."""
            if viz_type in ["All Rounds (Grid)", "Legacy Multi-Round"]:
                return gr.Slider(visible=False)
            else:
                return gr.Slider(visible=True)
        
        def get_summary():
            if not system.data_cache or "error" in system.data_cache:
                return "No results available yet. Run the pipeline to generate data."
            
            summary = "## 📊 Pipeline Results Summary\n\n"
            summary += f"**Total QIDs processed:** {len(system.data_cache)}\n\n"
            
            for qid, items in system.data_cache.items():
                summary += f"### QID: {qid[:32]}...\n"
                summary += f"- **Rounds completed:** {len(items)}\n"
                final_item = items[-1] if items else {}
                tests = final_item.get('synthesis_result', {}).get('tests') or []
                programs = final_item.get('gen_result', {}).get('eval_results') or []
                summary += f"- **Final state:** {len(programs)} programs × {len(tests)} test cases\n\n"
            
            return summary
        
        # Wire up events
        start_btn.click(
            start_pipeline,
            inputs=[rounds_slider, model_dropdown, max_tokens_slider, output_dir_text, overwrite_checkbox, api_key_text, backend_logging_checkbox],
            outputs=[status_text, log_display]
        )
        
        stop_btn.click(
            stop_pipeline,
            outputs=[status_text, log_display]
        )
        
        refresh_viz_btn.click(
            refresh_visualization,
            outputs=[qid_dropdown, matrix_plot, summary_text]
        )
        
        # Visualization type toggle
        viz_type_radio.change(
            toggle_round_slider,
            inputs=[viz_type_radio],
            outputs=[round_viz_slider]
        )
        
        # Update visualization when any control changes
        qid_dropdown.change(
            update_visualization,
            inputs=[qid_dropdown, viz_type_radio, round_viz_slider],
            outputs=[matrix_plot]
        )
        
        viz_type_radio.change(
            update_visualization,
            inputs=[qid_dropdown, viz_type_radio, round_viz_slider],
            outputs=[matrix_plot]
        )
        
        round_viz_slider.change(
            update_visualization,
            inputs=[qid_dropdown, viz_type_radio, round_viz_slider],
            outputs=[matrix_plot]
        )
        
        # Periodic status updates with auto-refresh visualization
        def periodic_update():
            status_msg, new_logs, progress = update_status()
            
            # Only return status updates, don't touch visualization unless explicitly needed
            return status_msg, new_logs
        
        # Set up automatic refresh every 2 seconds with status updates only
        status_timer = gr.Timer(2.0)
        status_timer.tick(
            periodic_update,
            outputs=[status_text, log_display]
        )
        
        # Manual visualization refresh after pipeline completion
        def load_final_results():
            """Load and display final results after pipeline completion."""
            print("📊 Loading final visualization results...")
            system.load_results()
            qid_choices = system.get_qid_list()
            
            if not qid_choices:
                empty_fig = go.Figure()
                empty_fig.add_annotation(text="No results found - please run the pipeline first", 
                                       xref="paper", yref="paper", x=0.5, y=0.5, 
                                       showarrow=False, font_size=16)
                return gr.Dropdown(choices=[], value=None), empty_fig, "No results available"
            
            # Load first QID by default with all rounds grid view
            first_qid = qid_choices[0]
            try:
                final_plot = system.create_all_rounds_matrices(first_qid)
                final_summary = get_summary()
                print(f"✅ Visualization loaded successfully for {len(qid_choices)} QIDs")
                return (
                    gr.Dropdown(choices=qid_choices, value=first_qid),
                    final_plot, 
                    final_summary
                )
            except Exception as e:
                print(f"❌ Error loading final results: {e}")
                error_fig = go.Figure()
                error_fig.add_annotation(text=f"Error: {str(e)}", 
                                       xref="paper", yref="paper", x=0.5, y=0.5, 
                                       showarrow=False, font_size=16)
                return (
                    gr.Dropdown(choices=qid_choices, value=first_qid),
                    error_fig,
                    f"Error loading visualization: {str(e)}"
                )
        
        load_results_btn.click(
            load_final_results,
            outputs=[qid_dropdown, matrix_plot, summary_text]
        )
    
    return app

if __name__ == "__main__":
    app = create_integrated_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        debug=False
    )
