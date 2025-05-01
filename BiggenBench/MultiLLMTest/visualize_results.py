import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize BigGen Bench benchmark results")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing result files")
    parser.add_argument("--score-file", type=str, default=None,
                        help="Specific score file to analyze (optional)")
    parser.add_argument("--output-dir", type=str, default="visualizations",
                        help="Directory to save visualizations")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"],
                        help="Output format for visualizations")
    return parser.parse_args()

def find_latest_score_file(results_dir):
    """Find the most recent score file in the results directory"""
    score_files = list(Path(results_dir).glob("*score_*.json"))
    if not score_files:
        raise FileNotFoundError(f"No score files found in {results_dir}")
    
    # Sort by modification time (most recent first)
    score_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return str(score_files[0])

def load_score_data(file_path):
    """Load score data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_overall_comparison(scores, output_dir, file_format="png"):
    """Create overall score comparison chart"""
    plt.figure(figsize=(12, 6))
    
    # Extract agent names and average scores
    agents = list(scores.keys())
    avg_scores = [scores[agent]["average_score"] for agent in agents]
    
    # Sort by average score (descending)
    sorted_indices = np.argsort(avg_scores)[::-1]
    agents = [agents[i] for i in sorted_indices]
    avg_scores = [avg_scores[i] for i in sorted_indices]
    
    # Create bar chart
    sns.set_style("whitegrid")
    ax = sns.barplot(x=agents, y=avg_scores, palette="viridis")
    
    # Add labels and title
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Average Score (out of 5)", fontsize=12)
    plt.title("Overall Model Performance Comparison", fontsize=14, fontweight="bold")
    
    # Add score values on top of bars
    for i, score in enumerate(avg_scores):
        ax.text(i, score + 0.1, f"{score:.2f}", ha="center", fontweight="bold")
    
    # Set y-axis limit
    plt.ylim(0, 5.5)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45 if len(agents) > 3 else 0, ha="right" if len(agents) > 3 else "center")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"overall_comparison.{file_format}")
    plt.savefig(output_path, dpi=300)
    print(f"Overall comparison chart saved to {output_path}")
    
    return output_path

def create_capability_comparison(scores, output_dir, file_format="png"):
    """Create comparison chart by capability"""
    # Extract all capabilities
    all_capabilities = set()
    for agent in scores:
        all_capabilities.update(scores[agent]["scores_by_capability"].keys())
    
    capabilities = sorted(list(all_capabilities))
    agents = list(scores.keys())
    
    # Prepare data for bar chart
    capability_scores = []
    for capability in capabilities:
        for agent in agents:
            if capability in scores[agent]["scores_by_capability"]:
                avg_score = scores[agent]["scores_by_capability"][capability]["average"]
                capability_scores.append({
                    "Capability": capability,
                    "Model": agent,
                    "Score": avg_score
                })
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Create grouped bar chart
    sns.set_style("whitegrid")
    ax = sns.barplot(
        x="Capability",
        y="Score",
        hue="Model",
        data=capability_scores,
        palette="viridis"
    )
    
    # Add labels and title
    plt.xlabel("Capability", fontsize=12)
    plt.ylabel("Average Score (out of 5)", fontsize=12)
    plt.title("Model Performance by Capability", fontsize=14, fontweight="bold")
    
    # Set y-axis limit
    plt.ylim(0, 5.5)
    
    # Adjust legend
    plt.legend(title="Model", loc="upper right", bbox_to_anchor=(1, 1))
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha="right")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"capability_comparison.{file_format}")
    plt.savefig(output_path, dpi=300)
    print(f"Capability comparison chart saved to {output_path}")
    
    return output_path

def create_task_comparison(scores, output_dir, file_format="png"):
    """Create comparison chart by task"""
    # Extract all tasks
    all_tasks = set()
    for agent in scores:
        all_tasks.update(scores[agent]["scores_by_task"].keys())
    
    tasks = sorted(list(all_tasks))
    agents = list(scores.keys())
    
    # Prepare data for heatmap
    task_matrix = np.zeros((len(agents), len(tasks)))
    
    for i, agent in enumerate(agents):
        for j, task in enumerate(tasks):
            if task in scores[agent]["scores_by_task"]:
                task_matrix[i, j] = scores[agent]["scores_by_task"][task]["average"]
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Create heatmap
    sns.set_style("white")
    ax = sns.heatmap(
        task_matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0,
        vmax=5,
        linewidths=.5,
        xticklabels=tasks,
        yticklabels=agents
    )
    
    # Add labels and title
    plt.xlabel("Task", fontsize=12)
    plt.ylabel("Model", fontsize=12)
    plt.title("Model Performance by Task", fontsize=14, fontweight="bold")
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha="right")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"task_comparison.{file_format}")
    plt.savefig(output_path, dpi=300)
    print(f"Task comparison chart saved to {output_path}")
    
    return output_path

def create_radar_chart(scores, output_dir, file_format="png"):
    """Create radar chart to compare models across capabilities"""
    # Extract all capabilities
    all_capabilities = set()
    for agent in scores:
        all_capabilities.update(scores[agent]["scores_by_capability"].keys())
    
    capabilities = sorted(list(all_capabilities))
    agents = list(scores.keys())
    
    # Prepare data for radar chart
    n = len(capabilities)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(capabilities)
    
    # Set y-axis limit
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"])
    
    # Add title
    plt.title("Model Performance Across Capabilities", fontsize=14, fontweight="bold", y=1.1)
    
    # Plot each agent
    for i, agent in enumerate(agents):
        agent_scores = []
        for capability in capabilities:
            if capability in scores[agent]["scores_by_capability"]:
                agent_scores.append(scores[agent]["scores_by_capability"][capability]["average"])
            else:
                agent_scores.append(0)
        
        # Close the polygon
        agent_scores += agent_scores[:1]
        
        # Plot agent scores
        ax.plot(angles, agent_scores, linewidth=2, label=agent)
        ax.fill(angles, agent_scores, alpha=0.1)
    
    # Add legend
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"radar_chart.{file_format}")
    plt.savefig(output_path, dpi=300)
    print(f"Radar chart saved to {output_path}")
    
    return output_path

def create_summary_table(scores, output_dir, file_format="png"):
    """Create summary table with scores"""
    # Prepare table data
    table_data = []
    
    # Extract all capabilities and tasks
    all_capabilities = set()
    all_tasks = set()
    
    for agent in scores:
        all_capabilities.update(scores[agent]["scores_by_capability"].keys())
        all_tasks.update(scores[agent]["scores_by_task"].keys())
    
    capabilities = sorted(list(all_capabilities))
    tasks = sorted(list(all_tasks))
    agents = sorted(list(scores.keys()))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8 + len(agents) * 0.4))
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ["Model", "Overall"] + capabilities
    
    for agent in agents:
        row = [agent, f"{scores[agent]['average_score']:.2f}"]
        for capability in capabilities:
            if capability in scores[agent]["scores_by_capability"]:
                row.append(f"{scores[agent]['scores_by_capability'][capability]['average']:.2f}")
            else:
                row.append("N/A")
        table_data.append(row)
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colColours=['#f0f0f0'] * len(headers),
        colWidths=[0.1] + [0.05] * (len(headers) - 1)
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Add title
    plt.suptitle("Summary of Model Performance", fontsize=14, fontweight="bold", y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"summary_table.{file_format}")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Summary table saved to {output_path}")
    
    return output_path

def create_html_report(score_file, visualization_paths, output_dir):
    """Create HTML report with all visualizations"""
    with open(score_file, 'r') as f:
        scores = json.load(f)
    
    # Extract metadata from filename
    filename = os.path.basename(score_file)
    timestamp = filename.split('_')[-1].split('.')[0]
    
    # Prepare HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BigGen Bench Benchmark Results</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .chart-container {{
                margin: 30px 0;
                text-align: center;
            }}
            img {{
                max-width: 100%;
                height: auto;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .score-highlight {{
                font-weight: bold;
                font-size: 1.2em;
            }}
            .model-card {{
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                margin: 20px 0;
                background-color: #f9f9f9;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>BigGen Bench Benchmark Results</h1>
            <p>Generated on: {timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[8:10]}:{timestamp[10:12]}:{timestamp[12:14]}</p>
            
            <h2>Overall Results</h2>
            <div class="chart-container">
                <img src="{os.path.basename(visualization_paths['overall'])}" alt="Overall Model Performance">
            </div>
            
            <h2>Results by Capability</h2>
            <div class="chart-container">
                <img src="{os.path.basename(visualization_paths['capability'])}" alt="Model Performance by Capability">
            </div>
            
            <h2>Results by Task</h2>
            <div class="chart-container">
                <img src="{os.path.basename(visualization_paths['task'])}" alt="Model Performance by Task">
            </div>
            
            <h2>Radar Comparison</h2>
            <div class="chart-container">
                <img src="{os.path.basename(visualization_paths['radar'])}" alt="Radar Chart Comparison">
            </div>
            
            <h2>Detailed Model Performance</h2>
    """
    
    # Add model details
    for agent, score_data in scores.items():
        html_content += f"""
            <div class="model-card">
                <h3>{agent}</h3>
                <p>Overall Score: <span class="score-highlight">{score_data['average_score']:.2f}/5.00</span></p>
                
                <h4>Performance by Capability</h4>
                <table>
                    <tr>
                        <th>Capability</th>
                        <th>Average Score</th>
                    </tr>
        """
        
        for capability, capability_data in score_data["scores_by_capability"].items():
            html_content += f"""
                    <tr>
                        <td>{capability}</td>
                        <td>{capability_data['average']:.2f}/5.00</td>
                    </tr>
            """
        
        html_content += """
                </table>
                
                <h4>Performance by Task</h4>
                <table>
                    <tr>
                        <th>Task</th>
                        <th>Average Score</th>
                    </tr>
        """
        
        for task, task_data in score_data["scores_by_task"].items():
            html_content += f"""
                    <tr>
                        <td>{task}</td>
                        <td>{task_data['average']:.2f}/5.00</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        """
    
    # Complete HTML
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, "benchmark_report.html")
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report saved to {html_path}")
    return html_path

def main():
    args = parse_args()
    
    # Find score file
    score_file = args.score_file
    if not score_file:
        score_file = find_latest_score_file(args.results_dir)
    
    print(f"Analyzing results from: {score_file}")
    
    # Load score data
    scores = load_score_data(score_file)
    
    # Create visualizations
    visualization_paths = {}
    visualization_paths['overall'] = create_overall_comparison(scores, args.output_dir, args.format)
    visualization_paths['capability'] = create_capability_comparison(scores, args.output_dir, args.format)
    visualization_paths['task'] = create_task_comparison(scores, args.output_dir, args.format)
    visualization_paths['radar'] = create_radar_chart(scores, args.output_dir, args.format)
    visualization_paths['summary'] = create_summary_table(scores, args.output_dir, args.format)
    
    # Create HTML report
    create_html_report(score_file, visualization_paths, args.output_dir)
    
    print("\nVisualization complete! You can find all outputs in:", args.output_dir)

if __name__ == "__main__":
    main()
