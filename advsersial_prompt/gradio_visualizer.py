#!/usr/bin/env python3
"""
🎨 Gradio-based Advanced Visualizer for Multi-round Adversarial Generation

This module provides an interactive web interface for visualizing the dynamic
evolution of programs and test cases across multiple rounds of adversarial generation.

Features:
- Interactive matrix visualization with Plotly
- Round-by-round comparison
- Statistical insights
- Modern UI with real-time updates
- Export capabilities
"""

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime

class AdvancedVisualizer:
    """Advanced visualization system for adversarial generation results."""
    
    def __init__(self, output_dir: str = "outputs/acecoder_rounds"):
        self.output_dir = Path(output_dir)
        self.history_file = self.output_dir / "visualizations" / "visualization_history.jsonl"
        self.data_cache = {}
        self.load_data()
    
    def load_data(self):
        """Load and process visualization data from history file."""
        if not self.history_file.exists():
            self.data_cache = {"error": "No history file found"}
            return
            
        try:
            history_data = []
            with open(self.history_file, 'r') as f:
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
            self.data_cache = {"error": f"Error loading data: {str(e)}"}
    
    def get_qid_list(self) -> List[str]:
        """Get list of available QIDs."""
        if "error" in self.data_cache:
            return []
        return list(self.data_cache.keys())
    
    def get_round_data(self, qid: str, round_num: int) -> Optional[Dict]:
        """Get data for a specific QID and round."""
        if qid not in self.data_cache:
            return None
        
        for item in self.data_cache[qid]:
            if item.get('round', 0) == round_num:
                return item
        return None
    
    def create_matrix_heatmap(self, qid: str, round_num: int) -> go.Figure:
        """Create an interactive heatmap for program vs test case matrix."""
        data = self.get_round_data(qid, round_num)
        if not data:
            fig = go.Figure()
            fig.add_annotation(text="No data available", xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False, font_size=16)
            return fig
        
        tests = data.get('synthesis_result', {}).get('tests') or []
        eval_results = data.get('gen_result', {}).get('eval_results') or []
        
        if not tests or not eval_results:
            fig = go.Figure()
            fig.add_annotation(text="No test or evaluation data", xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False, font_size=16)
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
                if test_idx == 0 and prog_idx == 0:  # Only add test labels once
                    test_labels.append(f"Test {test_idx + 1}")
                elif prog_idx == 0:
                    test_labels.append(f"Test {test_idx + 1}")
                
                if test_idx < len(test_cases_pass_status):
                    test_result = test_cases_pass_status[test_idx]
                    if isinstance(test_result, dict):
                        passed = test_result.get('pass', False)
                    else:
                        passed = bool(test_result)  # Handle boolean values
                    
                    row.append(2 if passed else 0)  # 🔧 FIX: Use 2 for pass, 0 for fail, 1 for N/A
                    status = "✅ PASS" if passed else "❌ FAIL"
                    
                    # Truncate test for hover
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
        
        round_type = "Generate Programs (Filter Tests)" if round_num % 2 == 1 else "Generate Tests (Filter Programs)"
        
        fig.update_layout(
            title=f"Round {round_num} Matrix - {round_type}<br>QID: {qid[:16]}...",
            xaxis_title="Test Cases",
            yaxis_title="Programs",
            font=dict(size=12),
            width=800,
            height=500
        )
        
        return fig
    
    def create_evolution_summary(self, qid: str) -> go.Figure:
        """Create a summary chart showing evolution across rounds."""
        if qid not in self.data_cache:
            fig = go.Figure()
            fig.add_annotation(text="QID not found", xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False, font_size=16)
            return fig
        
        rounds = []
        num_programs = []
        num_tests = []
        avg_pass_rates = []
        
        for item in self.data_cache[qid]:
            round_num = item.get('round', 0)
            tests = item.get('synthesis_result', {}).get('tests') or []
            eval_results = item.get('gen_result', {}).get('eval_results') or []
            
            rounds.append(round_num)
            num_programs.append(len(eval_results))
            num_tests.append(len(tests))
            
            # Calculate average pass rate
            if eval_results:
                pass_rates = [er.get('pass_rate', 0) for er in eval_results]
                avg_pass_rates.append(np.mean(pass_rates))
            else:
                avg_pass_rates.append(0)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Programs & Test Cases Count", 
                "Average Pass Rate Evolution",
                "Round Type Pattern",
                "Matrix Size Evolution"
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Counts
        fig.add_trace(
            go.Scatter(x=rounds, y=num_programs, name="Programs", line=dict(color="blue", width=3)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=rounds, y=num_tests, name="Test Cases", line=dict(color="red", width=3)),
            row=1, col=1
        )
        
        # Plot 2: Pass rates
        fig.add_trace(
            go.Scatter(x=rounds, y=avg_pass_rates, name="Avg Pass Rate", 
                      line=dict(color="green", width=3), mode="lines+markers"),
            row=1, col=2
        )
        
        # Plot 3: Round types
        round_types = ["Generate Programs" if r % 2 == 1 else "Generate Tests" for r in rounds]
        colors = ["lightblue" if rt == "Generate Programs" else "lightcoral" for rt in round_types]
        
        fig.add_trace(
            go.Bar(x=rounds, y=[1]*len(rounds), name="Round Type", 
                   marker_color=colors, text=round_types, textposition="inside"),
            row=2, col=1
        )
        
        # Plot 4: Matrix size (programs × tests)
        matrix_sizes = [p * t for p, t in zip(num_programs, num_tests)]
        fig.add_trace(
            go.Scatter(x=rounds, y=matrix_sizes, name="Matrix Size (P×T)", 
                      line=dict(color="purple", width=3), mode="lines+markers"),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text=f"Evolution Summary for QID: {qid[:32]}...",
            showlegend=True
        )
        
        return fig
    
    def get_qid_info(self, qid: str) -> str:
        """Get detailed information about a QID."""
        if qid not in self.data_cache:
            return "❌ QID not found"
        
        items = self.data_cache[qid]
        if not items:
            return "❌ No data for this QID"
        
        # Get problem description from first item
        first_item = items[0]
        problem = first_item.get('synthesis_result', {}).get('problem', 'N/A')
        
        total_rounds = len(items)
        
        # Calculate statistics
        final_item = items[-1]
        final_tests = final_item.get('synthesis_result', {}).get('tests') or []
        final_programs = final_item.get('gen_result', {}).get('eval_results') or []
        
        info = f"""
## 📋 QID Information

**QID:** `{qid}`

**Total Rounds:** {total_rounds}

**Final State:**
- Programs: {len(final_programs)}
- Test Cases: {len(final_tests)}

**Problem Description:**
```
{problem[:300]}{'...' if len(problem) > 300 else ''}
```

**Evolution Pattern:**
"""
        
        for i, item in enumerate(items):
            round_num = item.get('round', 0)
            tests = item.get('synthesis_result', {}).get('tests') or []
            programs = item.get('gen_result', {}).get('eval_results') or []
            round_type = "🔵 Generate Programs" if round_num % 2 == 1 else "🔴 Generate Tests"
            info += f"- Round {round_num}: {round_type} → {len(programs)}P × {len(tests)}T\n"
        
        return info
    
    def get_test_details(self, qid: str, round_num: int) -> str:
        """Get detailed test case information for a specific round."""
        data = self.get_round_data(qid, round_num)
        if not data:
            return "❌ No data available for this round"
        
        tests = data.get('synthesis_result', {}).get('tests') or []
        if not tests:
            return "❌ No test cases found"
        
        details = f"## 🧪 Test Cases for Round {round_num}\n\n"
        
        for i, test in enumerate(tests):
            details += f"### Test Case {i+1}\n```python\n{test}\n```\n\n"
        
        return details
    
    def get_program_details(self, qid: str, round_num: int) -> str:
        """Get detailed program information for a specific round."""
        data = self.get_round_data(qid, round_num)
        if not data:
            return "❌ No data available for this round"
        
        eval_results = data.get('gen_result', {}).get('eval_results') or []
        if not eval_results:
            return "❌ No program evaluation results found"
        
        details = f"## 💻 Program Evaluation for Round {round_num}\n\n"
        
        for i, result in enumerate(eval_results):
            pass_rate = result.get('pass_rate', 0)
            details += f"### Program {i+1}\n"
            details += f"- **Pass Rate:** {pass_rate:.2%}\n"
            
            # Show pass/fail status for each test
            test_status = result.get('test_cases_pass_status', [])
            if test_status:
                details += f"- **Test Results:** "
                for j, status in enumerate(test_status):
                    symbol = "✅" if status.get('pass', False) else "❌"
                    details += f"T{j+1}:{symbol} "
                details += "\n"
            
            details += "\n"
        
        return details

def create_gradio_interface():
    """Create and configure the Gradio interface."""
    
    # Initialize visualizer
    viz = AdvancedVisualizer()
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
    .gr-button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
    }
    .gr-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }
    """
    
    with gr.Blocks(
        title="🎨 Adversarial Generation Visualizer",
        theme=gr.themes.Soft(primary_hue="blue"),
        css=custom_css
    ) as app:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1>🎨 Advanced Adversarial Generation Visualizer</h1>
            <p>Interactive visualization of multi-round program-test case evolution</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3>🎯 Controls</h3>")
                
                qid_dropdown = gr.Dropdown(
                    choices=viz.get_qid_list(),
                    label="Select QID",
                    info="Choose a Question ID to visualize",
                    value=viz.get_qid_list()[0] if viz.get_qid_list() else None
                )
                
                round_slider = gr.Slider(
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=0,
                    label="Round Number",
                    info="Select which round to visualize"
                )
                
                refresh_btn = gr.Button("🔄 Refresh Data", variant="primary")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML("<h3>📊 Interactive Matrix Visualization</h3>")
                matrix_plot = gr.Plot(label="Program vs Test Case Matrix")
                
            with gr.Column(scale=1):
                gr.HTML("<h3>📈 Evolution Summary</h3>")
                evolution_plot = gr.Plot(label="Evolution Across Rounds")
        
        with gr.Row():
            with gr.Column():
                gr.HTML("<h3>📋 QID Information</h3>")
                qid_info = gr.Markdown(value="Select a QID to see information")
        
        with gr.Row():
            with gr.Column():
                gr.HTML("<h3>🧪 Test Case Details</h3>")
                test_details = gr.Markdown(value="Select a QID and round to see test cases")
                
            with gr.Column():
                gr.HTML("<h3>💻 Program Details</h3>")
                program_details = gr.Markdown(value="Select a QID and round to see programs")
        
        # Event handlers
        def update_visualizations(qid, round_num):
            if not qid:
                return None, None, "No QID selected", "No data", "No data"
            
            matrix_fig = viz.create_matrix_heatmap(qid, round_num)
            evolution_fig = viz.create_evolution_summary(qid)
            qid_info_text = viz.get_qid_info(qid)
            test_info = viz.get_test_details(qid, round_num)
            program_info = viz.get_program_details(qid, round_num)
            
            return matrix_fig, evolution_fig, qid_info_text, test_info, program_info
        
        def refresh_data():
            viz.load_data()
            new_choices = viz.get_qid_list()
            return gr.Dropdown(choices=new_choices, value=new_choices[0] if new_choices else None)
        
        # Wire up events
        qid_dropdown.change(
            update_visualizations,
            inputs=[qid_dropdown, round_slider],
            outputs=[matrix_plot, evolution_plot, qid_info, test_details, program_details]
        )
        
        round_slider.change(
            update_visualizations,
            inputs=[qid_dropdown, round_slider],
            outputs=[matrix_plot, evolution_plot, qid_info, test_details, program_details]
        )
        
        refresh_btn.click(
            refresh_data,
            outputs=[qid_dropdown]
        )
        
        # Initial load
        app.load(
            update_visualizations,
            inputs=[qid_dropdown, round_slider],
            outputs=[matrix_plot, evolution_plot, qid_info, test_details, program_details]
        )
    
    return app

if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        debug=True
    )
