#!/usr/bin/env python3
"""
Advanced analysis and visualization of chess probing results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import chess
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

def load_results(results_path: str) -> Dict:
    """Load probing results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)

def create_detailed_analysis(results: Dict) -> Dict:
    """Create detailed analysis of probing results."""
    analysis = {}
    
    for layer_key, layer_results in results.items():
        accuracies = list(layer_results.values())
        squares = list(layer_results.keys())
        
        # Basic statistics
        analysis[layer_key] = {
            'mean': np.mean(accuracies),
            'std': np.std(accuracies),
            'min': np.min(accuracies),
            'max': np.max(accuracies),
            'median': np.median(accuracies),
            'q25': np.percentile(accuracies, 25),
            'q75': np.percentile(accuracies, 75)
        }
        
        # Identify best and worst performing squares
        sorted_squares = sorted(zip(squares, accuracies), key=lambda x: x[1], reverse=True)
        analysis[layer_key]['best_squares'] = sorted_squares[:5]
        analysis[layer_key]['worst_squares'] = sorted_squares[-5:]
        
        # Categorize squares by performance
        high_acc = [s for s, a in sorted_squares if a > analysis[layer_key]['q75']]
        low_acc = [s for s, a in sorted_squares if a < analysis[layer_key]['q25']]
        
        analysis[layer_key]['high_performance_squares'] = high_acc
        analysis[layer_key]['low_performance_squares'] = low_acc
        
        # Chess-specific analysis
        analysis[layer_key]['chess_analysis'] = analyze_chess_patterns(layer_results)
    
    return analysis

def analyze_chess_patterns(square_results: Dict[str, float]) -> Dict:
    """Analyze chess-specific patterns in the results."""
    patterns = {}
    
    # Group by ranks and files
    rank_accuracies = {str(i): [] for i in range(1, 9)}
    file_accuracies = {f: [] for f in 'abcdefgh'}
    
    for square, accuracy in square_results.items():
        file = square[0]
        rank = square[1]
        
        file_accuracies[file].append(accuracy)
        rank_accuracies[rank].append(accuracy)
    
    # Calculate average accuracy by rank and file
    patterns['rank_averages'] = {r: np.mean(accs) for r, accs in rank_accuracies.items()}
    patterns['file_averages'] = {f: np.mean(accs) for f, accs in file_accuracies.items()}
    
    # Identify center vs edge patterns
    center_squares = ['d4', 'd5', 'e4', 'e5']
    edge_squares = [f + r for f in 'ah' for r in '18'] + [f + r for f in 'bcdefg' for r in '18']
    
    center_acc = [square_results[sq] for sq in center_squares if sq in square_results]
    edge_acc = [square_results[sq] for sq in edge_squares if sq in square_results]
    
    patterns['center_accuracy'] = np.mean(center_acc) if center_acc else 0
    patterns['edge_accuracy'] = np.mean(edge_acc) if edge_acc else 0
    
    # Color pattern analysis
    white_squares = []
    black_squares = []
    
    for square, accuracy in square_results.items():
        square_idx = chess.parse_square(square)
        is_light_square = chess.square_is_light(square_idx)
        
        if is_light_square:
            white_squares.append(accuracy)
        else:
            black_squares.append(accuracy)
    
    patterns['light_square_accuracy'] = np.mean(white_squares) if white_squares else 0
    patterns['dark_square_accuracy'] = np.mean(black_squares) if black_squares else 0
    
    return patterns

def create_comprehensive_visualizations(results: Dict, analysis: Dict, output_dir: str):
    """Create comprehensive visualizations of the results."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    colors = sns.color_palette("husl", len(results))
    
    # 1. Accuracy distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Histogram of accuracies
    for i, (layer_key, layer_results) in enumerate(results.items()):
        accuracies = list(layer_results.values())
        axes[0, 0].hist(accuracies, alpha=0.7, label=layer_key, bins=20, color=colors[i])
    
    axes[0, 0].set_xlabel('Accuracy')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Square Accuracies')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plot comparison
    box_data = []
    box_labels = []
    for layer_key, layer_results in results.items():
        box_data.append(list(layer_results.values()))
        box_labels.append(layer_key)
    
    axes[0, 1].boxplot(box_data, labels=box_labels)
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy Distribution by Layer')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Rank and file analysis
    layer_key = list(results.keys())[0]  # Use first layer for rank/file analysis
    chess_analysis = analysis[layer_key]['chess_analysis']
    
    # Rank averages
    ranks = list(chess_analysis['rank_averages'].keys())
    rank_accs = list(chess_analysis['rank_averages'].values())
    
    axes[1, 0].bar(ranks, rank_accs, color='skyblue', alpha=0.7)
    axes[1, 0].set_xlabel('Rank')
    axes[1, 0].set_ylabel('Average Accuracy')
    axes[1, 0].set_title('Average Accuracy by Rank')
    axes[1, 0].grid(True, alpha=0.3)
    
    # File averages
    files = list(chess_analysis['file_averages'].keys())
    file_accs = list(chess_analysis['file_averages'].values())
    
    axes[1, 1].bar(files, file_accs, color='lightcoral', alpha=0.7)
    axes[1, 1].set_xlabel('File')
    axes[1, 1].set_ylabel('Average Accuracy')
    axes[1, 1].set_title('Average Accuracy by File')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Layer comparison heatmaps
    if len(results) > 1:
        fig, axes = plt.subplots(1, len(results), figsize=(8 * len(results), 6))
        if len(results) == 1:
            axes = [axes]
        
        for i, (layer_key, layer_results) in enumerate(results.items()):
            # Create 8x8 matrix
            accuracy_matrix = np.zeros((8, 8))
            
            for square_name, accuracy in layer_results.items():
                square_idx = chess.parse_square(square_name)
                row = square_idx // 8
                col = square_idx % 8
                accuracy_matrix[7-row, col] = accuracy
            
            im = axes[i].imshow(accuracy_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
            axes[i].set_title(f'{layer_key}')
            axes[i].set_xticks(range(8))
            axes[i].set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
            axes[i].set_yticks(range(8))
            axes[i].set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])
            
            # Add accuracy values as text
            for row in range(8):
                for col in range(8):
                    text = axes[i].text(col, row, f'{accuracy_matrix[row, col]:.2f}',
                                       ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path / 'layer_comparison_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Chess pattern analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Center vs Edge comparison
    for i, (layer_key, layer_analysis) in enumerate(analysis.items()):
        chess_patterns = layer_analysis['chess_analysis']
        
        categories = ['Center', 'Edge', 'Light Squares', 'Dark Squares']
        values = [
            chess_patterns['center_accuracy'],
            chess_patterns['edge_accuracy'],
            chess_patterns['light_square_accuracy'],
            chess_patterns['dark_square_accuracy']
        ]
        
        x_pos = np.arange(len(categories))
        axes[i//2, i%2].bar(x_pos, values, alpha=0.7, color=colors[i])
        axes[i//2, i%2].set_xticks(x_pos)
        axes[i//2, i%2].set_xticklabels(categories, rotation=45)
        axes[i//2, i%2].set_ylabel('Average Accuracy')
        axes[i//2, i%2].set_title(f'Chess Pattern Analysis - {layer_key}')
        axes[i//2, i%2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'chess_pattern_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Performance ranking
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Best performing squares
    layer_key = list(results.keys())[0]
    best_squares = analysis[layer_key]['best_squares']
    worst_squares = analysis[layer_key]['worst_squares']
    
    best_names = [sq[0] for sq in best_squares]
    best_accs = [sq[1] for sq in best_squares]
    
    worst_names = [sq[0] for sq in worst_squares]
    worst_accs = [sq[1] for sq in worst_squares]
    
    axes[0].barh(range(len(best_names)), best_accs, color='green', alpha=0.7)
    axes[0].set_yticks(range(len(best_names)))
    axes[0].set_yticklabels(best_names)
    axes[0].set_xlabel('Accuracy')
    axes[0].set_title('Top 5 Best Performing Squares')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].barh(range(len(worst_names)), worst_accs, color='red', alpha=0.7)
    axes[1].set_yticks(range(len(worst_names)))
    axes[1].set_yticklabels(worst_names)
    axes[1].set_xlabel('Accuracy')
    axes[1].set_title('Top 5 Worst Performing Squares')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'performance_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_path}")

def generate_report(results: Dict, analysis: Dict, output_dir: str):
    """Generate a comprehensive text report."""
    output_path = Path(output_dir)
    
    report = []
    report.append("# Chess Model Probing Analysis Report")
    report.append("=" * 50)
    report.append("")
    
    # Overall summary
    report.append("## Overall Summary")
    report.append("")
    
    for layer_key, layer_analysis in analysis.items():
        report.append(f"### {layer_key}")
        report.append(f"- Mean accuracy: {layer_analysis['mean']:.3f}")
        report.append(f"- Standard deviation: {layer_analysis['std']:.3f}")
        report.append(f"- Minimum accuracy: {layer_analysis['min']:.3f}")
        report.append(f"- Maximum accuracy: {layer_analysis['max']:.3f}")
        report.append(f"- Median accuracy: {layer_analysis['median']:.3f}")
        report.append("")
    
    # Chess-specific patterns
    report.append("## Chess Pattern Analysis")
    report.append("")
    
    for layer_key, layer_analysis in analysis.items():
        chess_patterns = layer_analysis['chess_analysis']
        
        report.append(f"### {layer_key}")
        report.append(f"- Center squares accuracy: {chess_patterns['center_accuracy']:.3f}")
        report.append(f"- Edge squares accuracy: {chess_patterns['edge_accuracy']:.3f}")
        report.append(f"- Light squares accuracy: {chess_patterns['light_square_accuracy']:.3f}")
        report.append(f"- Dark squares accuracy: {chess_patterns['dark_square_accuracy']:.3f}")
        report.append("")
        
        # Best and worst squares
        report.append("#### Best performing squares:")
        for sq, acc in layer_analysis['best_squares']:
            report.append(f"  - {sq}: {acc:.3f}")
        report.append("")
        
        report.append("#### Worst performing squares:")
        for sq, acc in layer_analysis['worst_squares']:
            report.append(f"  - {sq}: {acc:.3f}")
        report.append("")
    
    # Save report
    with open(output_path / 'analysis_report.md', 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Report saved to {output_path / 'analysis_report.md'}")

def main():
    parser = argparse.ArgumentParser(description='Analyze chess probing results')
    parser.add_argument('--results_path', type=str, required=True,
                       help='Path to probing results JSON file')
    parser.add_argument('--output_dir', type=str, default='./analysis_output',
                       help='Directory to save analysis outputs')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results_path}")
    results = load_results(args.results_path)
    
    # Perform analysis
    print("Performing detailed analysis...")
    analysis = create_detailed_analysis(results)
    
    # Create visualizations
    print("Creating visualizations...")
    create_comprehensive_visualizations(results, analysis, args.output_dir)
    
    # Generate report
    print("Generating report...")
    generate_report(results, analysis, args.output_dir)
    
    # Print summary to console
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    
    for layer_key, layer_analysis in analysis.items():
        print(f"\n{layer_key}:")
        print(f"  Mean accuracy: {layer_analysis['mean']:.3f}")
        print(f"  Best square: {layer_analysis['best_squares'][0][0]} ({layer_analysis['best_squares'][0][1]:.3f})")
        print(f"  Worst square: {layer_analysis['worst_squares'][0][0]} ({layer_analysis['worst_squares'][0][1]:.3f})")
        
        chess_patterns = layer_analysis['chess_analysis']
        print(f"  Center vs Edge: {chess_patterns['center_accuracy']:.3f} vs {chess_patterns['edge_accuracy']:.3f}")
        print(f"  Light vs Dark: {chess_patterns['light_square_accuracy']:.3f} vs {chess_patterns['dark_square_accuracy']:.3f}")
    
    print(f"\nFull analysis saved to: {args.output_dir}")

if __name__ == '__main__':
    main()