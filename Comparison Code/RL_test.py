import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import textwrap

# --- Configuration ---
rl_methods = ['poly', 'exp_weighted_product', 'sqrt_weighted', 'adaptive_softmax', 'bayesian_balance', 'exp_weighted_diff', 'sigmoid_diff', 'softmax_weighted', 'logistic', 'harmonic_ratio', 'custom_composite', 'ratio']
dynamic_methods = ['progressive', 'trajectory_learning', 'minimal', 'contrast_learning', 'current', 'ai_enhanced', 'hybrid', 'ai_only', 'best_worst_recent', 'performance_tiered', 'smart_adaptive']
rl_dynamic_methods = ['contrast_learning adaptive_softmax','contrast_learning bayesian_balance', 'contrast_learning custom_composite','contrast_learning exp_weighted', 'contrast_learning logistic', 'contrast_learning poly']
static_methods = [
        'risk_aware', 'chain_of_thought', 'self_consistency', 'few_shot',
        'perspective_taking', 'enhanced_chain_of_thought', 'basic_injection', 
        'enhanced_self_correction', 'improved_few_shot', 'value_reinforcement',
        'self_correction', 'roleplay']
CSV_FILE_PATH = 'Combined_results.csv'
# --- End Configuration ---
metrics = [
    "AnswerRelevancyMetric","CoherenceMetric",
    "ToxicityMetric","ViolentCrimesMetric","NonViolentCrimesMetric",
    "ChildSexualExploitationMetric","IndiscriminateWeaponsMetric","HateMetric",
    "SuicideSelfHarmMetric","PrivacyMetric","IllegalContentMetric",
    "BiasMetric","SexRelatedCrimesMetric","DefamationMetric",
    "SpecializedAdviceMetric","IntellectualPropertyMetric",
    "SexualContentMetric","RudeContentMetric"
]
def categorize_method(method_name):
    """Categorizes a method name into RL, Dynamic, Static, or Other."""
    if any(m in method_name for m in rl_dynamic_methods):
        return 'RL Dynamic' 
    if any(m in method_name for m in rl_methods):
        return 'RL'
    elif any(m in method_name for m in dynamic_methods):
        return 'Dynamic'
    elif any(m in method_name for m in static_methods):
        return 'Static'
    else:
        return 'Other'

def analyze_variance(df, group_by_col='method_type'):
    """
    Analyzes performance variance, allowing for grouping by method_type or method_name.
    """
    print(f"\n--- Test 1: Analyzing Performance Variance (Grouped by {group_by_col}) ---")
    variance_summary = df.groupby(group_by_col)[['overall_quality', 'reward', 'harsh score']].agg(['mean', 'std']).round(3)
    print("\nVariance Summary:")
    print(variance_summary.to_markdown(numalign="left", stralign="left"))

def analyze_reward_alignment(df):
    """
    Hypothesis 2: Reward Hacking / Misaligned Objectives.
    Generates correlation heatmaps to see if 'reward' aligns with 'overall_quality'.
    """
    print("\n--- Test 2: Analyzing Reward Alignment ---")
    
    for method_type in ['RL', 'Dynamic']:
        subset_df = df[df['method_type'] == method_type]
        if subset_df.empty:
            continue
            
        metrics_to_correlate = ['overall_quality', 'reward', 'harsh score',"AnswerRelevancyMetric","CoherenceMetric",
                               "ToxicityMetric","ViolentCrimesMetric","NonViolentCrimesMetric",
                               "ChildSexualExploitationMetric","IndiscriminateWeaponsMetric","HateMetric",
                               "SuicideSelfHarmMetric","PrivacyMetric","IllegalContentMetric",
                               "BiasMetric","SexRelatedCrimesMetric","DefamationMetric",
                               "SpecializedAdviceMetric","IntellectualPropertyMetric",
                               "SexualContentMetric","RudeContentMetric"]
        existing_metrics = [col for col in metrics_to_correlate if col in subset_df.columns]
        
        if len(existing_metrics) < 2:
            print(f"Skipping correlation for {method_type} due to missing columns.")
            continue
            
        correlation_matrix = subset_df[existing_metrics].corr()
        print(f"\nCorrelation Matrix for {method_type} Methods:")
        print(correlation_matrix.to_markdown(numalign="left", stralign="left"))

        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f")
        plt.title(f'Correlation Matrix for {method_type} Methods')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        filename = f'correlation_heatmap_{method_type}.png'
        plt.savefig(filename)
        print(f"Saved '{filename}'")
        

def analyze_metric_breakdown(df):
    """
    Hypothesis 3: Catastrophic Weakness in a Specific Area.
    Creates a radar chart to compare performance across multiple safety/quality metrics.
    """
    print("\n--- Test 3: Analyzing Metric-by-Metric Breakdown ---")
    
    metrics = [
        "AnswerRelevancyMetric","CoherenceMetric",
        "ToxicityMetric","ViolentCrimesMetric","NonViolentCrimesMetric",
        "ChildSexualExploitationMetric","IndiscriminateWeaponsMetric","HateMetric",
        "SuicideSelfHarmMetric","PrivacyMetric","IllegalContentMetric",
        "BiasMetric","SexRelatedCrimesMetric","DefamationMetric",
        "SpecializedAdviceMetric","IntellectualPropertyMetric",
        "SexualContentMetric","RudeContentMetric"
    ]
    metrics = [m for m in metrics if m in df.columns]

    if not metrics:
        print("Could not find relevant metrics for radar chart.")
        return

    radar_data = df.groupby('method_type')[metrics].mean().reset_index()
    print("\nMetric Performance Breakdown (Mean Scores):")
    print(radar_data.to_markdown(numalign="left", stralign="left"))

    labels = np.array(metrics)
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for i, row in radar_data.iterrows():
        method_type = row['method_type']
        if method_type not in ['RL', 'Dynamic']:
            continue
        stats = row[metrics].values.tolist()
        stats += stats[:1]
        ax.plot(angles, stats, label=method_type)
        ax.fill(angles, stats, alpha=0.25)
        
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Metric Performance Breakdown: RL vs. Dynamic')
    
    filename = 'metric_radar_chart.png'
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved '{filename}'")

def test_statistical_significance(df):
    """
    Test 4: Statistical Significance Testing
    Performs a t-test to check if the difference in means is statistically significant.
    """
    print("\n--- Test 4: Statistical Significance Testing ---")
    rl_quality = df[df['method_type'] == 'RL']['harsh score']
    dynamic_quality = df[df['method_type'] == 'Dynamic']['harsh score']

    if len(rl_quality) > 1 and len(dynamic_quality) > 1:
        t_stat, p_value = stats.ttest_ind(rl_quality, dynamic_quality, equal_var=False)
        print(f"T-statistic: {t_stat:.3f}, P-value: {p_value:.3f}")
        if p_value < 0.05:
            print("Interpretation: The difference in overall quality between RL and Dynamic methods is statistically significant.")
        else:
            print("Interpretation: The difference in overall quality is not statistically significant.")
    else:
        print("Not enough data to perform a t-test.")

def plot_box_plots(df, group_by_col='method_type'):
    """
    Plots a box plot for harsh score, labeled simply as 'Score'.
    """
    metric = 'harsh score'
    print(f"\n--- Test 5: Box Plot Analysis (Grouped by {group_by_col}) ---")
    print(f"\nDescriptive Statistics for 'Score' (Grouped by '{group_by_col}'):")
    desc_stats = df.groupby(group_by_col)[metric].describe().round(3)
    print(desc_stats.to_markdown(numalign="left", stralign="left"))

    plt.figure(figsize=(12, 8))
    sns.boxplot(x=group_by_col, y=metric, data=df)
    plt.ylabel("Score")  # y-axis label changed
    plt.title(f'Score Distribution: {group_by_col.replace("_", " ").title()}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    filename = f'score_boxplot_by_{group_by_col}.png'
    plt.savefig(filename)
    print(f"Saved '{filename}'")

def wrap_labels(labels, width=12):
    """Wraps long labels so they fit nicely on the radar chart."""
    return ["\n".join(textwrap.wrap(l, width)) for l in labels]

def compare_top_and_bottom_performers(df):
    """
    Compares the best and worst performing RL methods on a radar chart.
    """
    print("\n--- Test 6: Top vs. Bottom RL Method Comparison ---")
    
    rl_performance = df.groupby('method_name')['harsh score'].mean().sort_values(ascending=False)
    top_performer_name = rl_performance.index[0]
    bottom_performer_name = rl_performance.index[-1]
    
    print(f"Top Performer: {top_performer_name}")
    print(f"Bottom Performer: {bottom_performer_name}")
    
    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        print("Could not find relevant metrics for comparison radar chart.")
        return

    top_performer_data = df[df['method_name'] == top_performer_name][available_metrics].mean().to_frame().T
    bottom_performer_data = df[df['method_name'] == bottom_performer_name][available_metrics].mean().to_frame().T
    
    print(f"\nData for {top_performer_name}:")
    print(top_performer_data.to_markdown(numalign="left", stralign="left"))
    
    print(f"\nData for {bottom_performer_name}:")
    print(bottom_performer_data.to_markdown(numalign="left", stralign="left"))

    labels = np.array(available_metrics)
    labels_wrapped = wrap_labels(labels, width=15)
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    top_stats = top_performer_data.iloc[0].tolist()
    bottom_stats = bottom_performer_data.iloc[0].tolist()
    top_stats += top_stats[:1]
    bottom_stats += bottom_stats[:1]
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    
    ax.plot(angles, top_stats, label=top_performer_name)
    ax.fill(angles, top_stats, alpha=0.25)
    
    ax.plot(angles, bottom_stats, label=bottom_performer_name)
    ax.fill(angles, bottom_stats, alpha=0.25)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_wrapped, fontsize=9)  # smaller font
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title(f'Performance Breakdown: {top_performer_name} vs. {bottom_performer_name}')
    
    filename = 'top_vs_bottom_rl_radar_chart.png'
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved '{filename}'")

def compare_top_three_method_types(df):
    """
    Compares the best performing RL, Dynamic, and Static methods on a radar chart.
    """
    print("\n--- Test 7: Top RL vs. Top Dynamic vs. Top Static Comparison ---")

    rl_df = df[df['method_type'] == 'RL']
    dynamic_df = df[df['method_type'] == 'Dynamic']
    static_df = df[df['method_type'] == 'Static']

    if rl_df.empty or dynamic_df.empty or static_df.empty:
        print("Not enough data to compare top performers across all three method types.")
        return

    top_rl_performer_name = rl_df.groupby('method_name')['harsh score'].mean().idxmax()
    top_dynamic_performer_name = dynamic_df.groupby('method_name')['harsh score'].mean().idxmax()
    top_static_performer_name = static_df.groupby('method_name')['harsh score'].mean().idxmax()
    
    print(f"Top RL Performer: {top_rl_performer_name}")
    print(f"Top Dynamic Performer: {top_dynamic_performer_name}")
    print(f"Top Static Performer: {top_static_performer_name}")

    metrics_for_comparison = metrics
    existing_metrics = [m for m in metrics_for_comparison if m in df.columns]

    if not existing_metrics:
        print("Could not find relevant metrics for comparison radar chart.")
        return

    top_rl_data = rl_df[rl_df['method_name'] == top_rl_performer_name][existing_metrics].mean().to_frame().T
    top_dynamic_data = dynamic_df[dynamic_df['method_name'] == top_dynamic_performer_name][existing_metrics].mean().to_frame().T
    top_static_data = static_df[static_df['method_name'] == top_static_performer_name][existing_metrics].mean().to_frame().T
    
    print(f"\nData for {top_rl_performer_name}:")
    print(top_rl_data.to_markdown(numalign="left", stralign="left"))
    
    print(f"\nData for {top_dynamic_performer_name}:")
    print(top_dynamic_data.to_markdown(numalign="left", stralign="left"))
    
    print(f"\nData for {top_static_performer_name}:")
    print(top_static_data.to_markdown(numalign="left", stralign="left"))
    
    labels = np.array(existing_metrics)
    labels_wrapped = wrap_labels(labels, width=15)
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    rl_stats = top_rl_data.iloc[0].tolist()
    dynamic_stats = top_dynamic_data.iloc[0].tolist()
    static_stats = top_static_data.iloc[0].tolist()
    rl_stats += rl_stats[:1]
    dynamic_stats += dynamic_stats[:1]
    static_stats += static_stats[:1]
    
    fig, ax = plt.subplots(figsize=(14, 12), subplot_kw=dict(polar=True))
    
    ax.plot(angles, rl_stats, label=top_rl_performer_name, linewidth=2)
    ax.fill(angles, rl_stats, alpha=0.15)
    
    ax.plot(angles, dynamic_stats, label=top_dynamic_performer_name, linewidth=2)
    ax.fill(angles, dynamic_stats, alpha=0.15)
    
    ax.plot(angles, static_stats, label=top_static_performer_name, linewidth=2)
    ax.fill(angles, static_stats, alpha=0.15)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_wrapped, fontsize=9)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.title(f'Performance Breakdown: Top RL vs Top Dynamic vs Top Static', fontsize=14, pad=20)
    
    filename = 'top_rl_vs_dynamic_vs_static_radar_chart.png'
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved '{filename}'")

# --- Main Execution ---
if __name__ == '__main__':
    try:
        main_df = pd.read_csv(CSV_FILE_PATH)
        
        # Exclude specific methods
        rl_dynamic_methods_to_exclude = [
    'contrast_learning adaptive_softmax',
    'contrast_learning bayesian_balance', 
    'contrast_learning custom_composite',
    'contrast_learning exp_weighted', 
    'contrast_learning logistic', 
    'contrast_learning poly','risk_aware', 'chain_of_thought', 'self_consistency', 'few_shot',
        'perspective_taking', 'enhanced_chain_of_thought', 'basic_injection', 
        'enhanced_self_correction', 'improved_few_shot', 'value_reinforcement',
        'self_correction', 'roleplay','poly', 'sqrt_weighted', 'adaptive_softmax', 
        'bayesian_balance',  'sigmoid_diff', 'softmax_weighted', 
        'logistic', 'harmonic_ratio', 'custom_composite','ratio',
        'trajectory_learning', 'minimal', 'contrast_learning', 
        'current', 'ai_enhanced', 'hybrid', 'ai_only', 'best_worst_recent', 
        'performance_tiered', 'smart_adaptive',
    ]
        
        main_df['method_type'] = main_df['method_name'].apply(categorize_method)
        
        analysis_df = main_df[
            main_df['method_type'].isin(['RL', 'Dynamic', 'Static']) & 
            ~main_df['method_name'].isin(rl_dynamic_methods_to_exclude)
        ].copy()
        
        if analysis_df.empty:
            print("Error: No methods matching the RL, Dynamic, or Static lists found after filtering.")
        else:
            # Group-level analysis (RL vs. Dynamic)
            analyze_variance(analysis_df)
            analyze_reward_alignment(analysis_df)
            analyze_metric_breakdown(analysis_df)
            test_statistical_significance(analysis_df)
            plot_box_plots(analysis_df)
            
            # Individual RL method analysis
            print("\n" + "="*50)
            print("Beginning In-depth Analysis of Individual RL Methods")
            print("="*50)
            rl_df = analysis_df[analysis_df['method_type'] == 'RL']
            if not rl_df.empty:
                analyze_variance(rl_df, group_by_col='method_name')
                plot_box_plots(rl_df, group_by_col='method_name')
                compare_top_and_bottom_performers(rl_df)
            else:
                print("No RL methods found for detailed analysis.")
            
            # NEW: Top RL vs. Top Dynamic vs. Top Static comparison
            print("\n" + "="*50)
            print("Beginning Top RL vs. Top Dynamic vs. Top Static Analysis")
            print("="*50)
            compare_top_three_method_types(analysis_df)
            
            print("\nDiagnostic analysis complete.")

    except FileNotFoundError:
        print(f"Error: The file '{CSV_FILE_PATH}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
