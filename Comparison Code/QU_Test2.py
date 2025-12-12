import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
import textwrap

sns.set(style='whitegrid', font_scale=1.0)

# ============================================================================
# STATISTICAL UTILITIES (from benchmark code)
# ============================================================================

def calculate_confidence_intervals(scores, confidence=0.95, n_bootstrap=1000):
    """Calculate bootstrap confidence intervals."""
    if len(scores) < 2:
        return (0.0, 0.0)
    
    bootstrap_means = []
    n = len(scores)
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return lower, upper

def compute_cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    if len(group1) < 2 or len(group2) < 2:
        return 0.0
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    
    if pooled_std == 0:
        return 0.0
    
    return (mean1 - mean2) / pooled_std

def perform_statistical_tests(df, baseline_method='no_system', metric='harsh score'):
    """Perform statistical comparisons between methods."""
    results = []
    
    methods = [m for m in df['method_name'].unique() if m != baseline_method]
    baseline_scores = df[df['method_name'] == baseline_method][metric].values
    
    if len(baseline_scores) < 2:
        print(f"Warning: Insufficient baseline data for {baseline_method}")
        return pd.DataFrame()
    
    for method in methods:
        method_scores = df[df['method_name'] == method][metric].values
        
        if len(method_scores) < 2:
            continue
        
        # T-test
        t_stat, p_value = stats.ttest_ind(method_scores, baseline_scores)
        
        # Effect size
        cohens_d = compute_cohens_d(method_scores, baseline_scores)
        
        # Confidence intervals
        method_ci = calculate_confidence_intervals(method_scores)
        
        # Effect size category
        if abs(cohens_d) < 0.2:
            effect_size = 'negligible'
        elif abs(cohens_d) < 0.5:
            effect_size = 'small'
        elif abs(cohens_d) < 0.8:
            effect_size = 'medium'
        else:
            effect_size = 'large'
        
        results.append({
            'method': method,
            'n_samples': len(method_scores),
            'mean': np.mean(method_scores),
            'std': np.std(method_scores),
            'ci_lower': method_ci[0],
            'ci_upper': method_ci[1],
            'vs_baseline_diff': np.mean(method_scores) - np.mean(baseline_scores),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01,
            'cohens_d': cohens_d,
            'effect_size': effect_size
        })
    
    return pd.DataFrame(results).sort_values('mean', ascending=False)

def perform_pairwise_comparisons(df, methods_to_compare, metric='harsh score'):
    """Perform pairwise statistical tests between specific methods."""
    results = []
    
    for i, method1 in enumerate(methods_to_compare):
        for method2 in methods_to_compare[i+1:]:
            scores1 = df[df['method_name'] == method1][metric].values
            scores2 = df[df['method_name'] == method2][metric].values
            
            if len(scores1) < 2 or len(scores2) < 2:
                continue
            
            # T-test
            t_stat, p_value = stats.ttest_ind(scores1, scores2)
            
            # Effect size
            cohens_d = compute_cohens_d(scores1, scores2)
            
            if abs(cohens_d) < 0.2:
                effect_size = 'negligible'
            elif abs(cohens_d) < 0.5:
                effect_size = 'small'
            elif abs(cohens_d) < 0.8:
                effect_size = 'medium'
            else:
                effect_size = 'large'
            
            results.append({
                'method1': method1,
                'method2': method2,
                'mean1': np.mean(scores1),
                'mean2': np.mean(scores2),
                'mean_diff': np.mean(scores1) - np.mean(scores2),
                't_statistic': t_stat,
                'p_value': p_value,
                'significant_at_0.05': p_value < 0.05,
                'significant_at_0.01': p_value < 0.01,
                'cohens_d': cohens_d,
                'effect_size': effect_size
            })
    
    return pd.DataFrame(results)

def categorize_method(method_name):
    """Categorize methods into groups."""
    method = method_name.lower()
    
    rl_dynamic_methods = [
        'contrast_learning adaptive_softmax',
        'contrast_learning bayesian_balance', 
        'contrast_learning custom_composite',
        'contrast_learning exp_weighted', 
        'contrast_learning logistic', 
        'contrast_learning poly'
    ]
    if any(rl_dynamic_method in method for rl_dynamic_method in rl_dynamic_methods):
        return 'RL Dynamic'  
    
    rl_methods = [
        'poly', 'exp_weighted_product', 'sqrt_weighted', 'adaptive_softmax', 
        'bayesian_balance', 'exp_weighted_diff', 'sigmoid_diff', 'softmax_weighted', 
        'logistic', 'harmonic_ratio', 'custom_composite','ratio'
    ]
    if any(rl_method in method for rl_method in rl_methods):
        return 'RL'
    
    static_methods = [
        'risk_aware', 'chain_of_thought', 'self_consistency', 'few_shot',
        'perspective_taking', 'enhanced_chain_of_thought', 'basic_injection', 
        'enhanced_self_correction', 'improved_few_shot', 'value_reinforcement',
        'self_correction', 'roleplay'
    ]
    if any(static_method in method for static_method in static_methods):
        return 'Static'
    
    dynamic_methods = [
        'progressive', 'trajectory_learning', 'minimal', 'contrast_learning', 
        'current', 'ai_enhanced', 'hybrid', 'ai_only', 'best_worst_recent', 
        'performance_tiered', 'smart_adaptive','raw_history'
    ]
    if any(dynamic_method in method for dynamic_method in dynamic_methods):
        return 'Dynamic'
    
    if method in ['none', 'no_system']:
        return 'Baseline'
    
    return 'Other'

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("="*80)
print("STATISTICAL ANALYSIS OF COMBINED RESULTS")
print("="*80 + "\n")

# Load data
df = pd.read_csv("Combined_results.csv")
df['model_name'] = df['model_name'].apply(lambda x: x.split('/')[-1] if isinstance(x, str) else x)

# exclude rude 
df = df[df['method_name'] != 'rude']
methods_to_exclude = [
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
        'trajectory_learning',
    ]

df = df[~df['method_name'].isin(methods_to_exclude)]



# Add method categories
df['method_group'] = df['method_name'].apply(categorize_method)

# Define metrics
metrics = [
    "AnswerRelevancyMetric","CoherenceMetric",
    "ToxicityMetric","ViolentCrimesMetric","NonViolentCrimesMetric",
    "ChildSexualExploitationMetric","IndiscriminateWeaponsMetric","HateMetric",
    "SuicideSelfHarmMetric","PrivacyMetric","IllegalContentMetric",
    "BiasMetric","SexRelatedCrimesMetric","DefamationMetric",
    "SpecializedAdviceMetric","IntellectualPropertyMetric",
    "SexualContentMetric","RudeContentMetric"
]

Q_metrics = ["AnswerRelevancyMetric", "BiasMetric", "CoherenceMetric"]
U_metrics = [m for m in metrics if m not in Q_metrics]

df = df.dropna(subset=metrics + ["harsh score"]).reset_index(drop=True)

print("DATASET OVERVIEW")
print("-"*80)
print(f"Total samples: {len(df)}")
print(f"Number of models: {df['model_name'].nunique()}")
print(f"Models: {df['model_name'].unique().tolist()}")
print(f"Number of methods: {df['method_name'].nunique()}")
print(f"Harsh score range: {df['harsh score'].min():.3f} to {df['harsh score'].max():.3f}")
print()

# ============================================================================
# STATISTICAL TESTS BY MODEL
# ============================================================================

print("="*80)
print("STATISTICAL COMPARISON TO BASELINE (PER MODEL)")
print("="*80 + "\n")

all_model_stats = []

for model in sorted(df['model_name'].unique()):
    print(f"\n{'='*60}")
    print(f"MODEL: {model}")
    print(f"{'='*60}\n")
    
    model_df = df[df['model_name'] == model]
    
    # Perform statistical tests
    stats_results = perform_statistical_tests(model_df, baseline_method='no_system', metric='harsh score')
    
    if stats_results.empty:
        print("Insufficient data for statistical analysis")
        continue
    
    stats_results['model'] = model
    all_model_stats.append(stats_results)
    
    # Display results
    print(f"{'Method':<25} {'Mean':<8} {'±Std':<8} {'95% CI':<20} {'vs Base':<10} {'p-value':<10} {'Sig':<5} {'Cohen d':<10} {'Effect':<12}")
    print("-"*120)
    
    for _, row in stats_results.iterrows():
        sig_marker = "***" if row['significant_at_0.01'] else ("**" if row['significant_at_0.05'] else "ns")
        ci_str = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
        
        print(f"{row['method']:<25} {row['mean']:>6.3f}  {row['std']:>6.3f}  {ci_str:<20} {row['vs_baseline_diff']:>8.3f}  "
              f"{row['p_value']:>8.4f}  {sig_marker:<5} {row['cohens_d']:>8.3f}  {row['effect_size']:<12}")
    
    # Top 5 methods
    print(f"\n🏆 TOP 5 METHODS (by mean harsh score):")
    print("-"*60)
    top5 = stats_results.head(5)
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        sig = "✓" if row['significant_at_0.05'] else "✗"
        print(f"  {i}. {row['method']:<30} {row['mean']:.3f} (p={row['p_value']:.4f}) {sig}")

# Combine all model statistics
if all_model_stats:
    combined_stats = pd.DataFrame(pd.concat(all_model_stats, ignore_index=True))
    
    # ============================================================================
    # CROSS-MODEL ANALYSIS
    # ============================================================================
    
    print("\n" + "="*80)
    print("CROSS-MODEL METHOD PERFORMANCE")
    print("="*80 + "\n")
    
    method_summary = combined_stats.groupby('method').agg({
        'mean': ['mean', 'std', 'count'],
        'vs_baseline_diff': 'mean',
        'cohens_d': 'mean',
        'significant_at_0.05': 'sum'
    }).round(4)
    
    method_summary.columns = ['avg_harsh_score', 'std_harsh_score', 'n_models', 
                              'avg_vs_baseline', 'avg_cohens_d', 'n_significant']
    method_summary = method_summary.sort_values('avg_harsh_score', ascending=False)
    
    print(f"{'Method':<30} {'Avg Score':<10} {'±Std':<8} {'N':<4} {'vs Base':<10} {'Cohen d':<10} {'Sig':<5}")
    print("-"*90)
    for method, row in method_summary.iterrows():
        print(f"{method:<30} {row['avg_harsh_score']:>8.3f}  {row['std_harsh_score']:>6.3f}  {row['n_models']:>2.0f}  "
              f"{row['avg_vs_baseline']:>8.3f}  {row['avg_cohens_d']:>8.3f}  {row['n_significant']:>3.0f}")
    
    # ============================================================================
    # METHOD GROUP ANALYSIS
    # ============================================================================
    
    print("\n" + "="*80)
    print("METHOD GROUP COMPARISON")
    print("="*80 + "\n")
    
    # Add method groups
    combined_stats['method_group'] = combined_stats['method'].apply(categorize_method)
    
    group_summary = combined_stats.groupby('method_group').agg({
        'mean': ['mean', 'std', 'count'],
        'vs_baseline_diff': 'mean',
        'cohens_d': 'mean',
        'significant_at_0.05': 'sum'
    }).round(4)
    
    group_summary.columns = ['avg_harsh_score', 'std_harsh_score', 'n_methods', 
                            'avg_vs_baseline', 'avg_cohens_d', 'n_significant']
    group_summary = group_summary.sort_values('avg_harsh_score', ascending=False)
    
    print(f"{'Group':<20} {'Avg Score':<10} {'±Std':<8} {'N':<4} {'vs Base':<10} {'Cohen d':<10} {'Sig':<5}")
    print("-"*80)
    for group, row in group_summary.iterrows():
        print(f"{group:<20} {row['avg_harsh_score']:>8.3f}  {row['std_harsh_score']:>6.3f}  {row['n_methods']:>2.0f}  "
              f"{row['avg_vs_baseline']:>8.3f}  {row['avg_cohens_d']:>8.3f}  {row['n_significant']:>3.0f}")
    
    # ============================================================================
    # PAIRWISE COMPARISONS (TOP METHODS)
    # ============================================================================
    
    print("\n" + "="*80)
    print("PAIRWISE COMPARISONS (TOP 10 METHODS)")
    print("="*80 + "\n")
    
    top_methods = method_summary.head(10).index.tolist()
    
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        
        # Filter to methods that exist in this model
        available_methods = [m for m in top_methods if m in model_df['method_name'].values]
        
        if len(available_methods) < 2:
            continue
        
        print(f"\n{model}:")
        print("-"*80)
        
        pairwise_results = perform_pairwise_comparisons(model_df, available_methods, metric='harsh score')
        
        if pairwise_results.empty:
            print("  Insufficient data for pairwise comparisons")
            continue
        
        # Show only significant comparisons
        significant = pairwise_results[pairwise_results['significant_at_0.05']]
        
        if significant.empty:
            print("  No statistically significant differences between top methods")
        else:
            print(f"  {'Method 1':<20} {'vs':<3} {'Method 2':<20} {'Diff':<8} {'p-value':<10} {'Effect':<12}")
            print("  " + "-"*75)
            for _, row in significant.iterrows():
                sig_marker = "***" if row['significant_at_0.01'] else "**"
                print(f"  {row['method1']:<20} vs  {row['method2']:<20} {row['mean_diff']:>6.3f}  "
                      f"{row['p_value']:>8.4f} {sig_marker}  {row['effect_size']:<12}")
    
    # ============================================================================
    # SAVE RESULTS
    # ============================================================================
    
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save all statistics
    combined_stats.to_csv("statistical_analysis_by_model.csv", index=False)
    print("✅ Saved: statistical_analysis_by_model.csv")
    
    method_summary.to_csv("statistical_analysis_method_summary.csv")
    print("✅ Saved: statistical_analysis_method_summary.csv")
    
    group_summary.to_csv("statistical_analysis_group_summary.csv")
    print("✅ Saved: statistical_analysis_group_summary.csv")
    
    # ============================================================================
    # VISUALIZATION
    # ============================================================================
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # 1. Method comparison with confidence intervals
    plt.figure(figsize=(14, max(8, len(method_summary) * 0.4)))

    y_pos = np.arange(len(method_summary))
    methods = method_summary.index.tolist()
    means = method_summary['avg_harsh_score'].values
    stds = method_summary['std_harsh_score'].values

    # Calculate CIs for each method
    ci_lower = []
    ci_upper = []
    for i, method in enumerate(methods):
        scores = df[df['method_name'] == method]['harsh score'].values
        ci_low, ci_high = calculate_confidence_intervals(scores)
        ci_lower.append(means[i] - ci_low)
        ci_upper.append(ci_high - means[i])

    plt.barh(y_pos, means, xerr=[ci_lower, ci_upper], alpha=0.7, capsize=5)
    plt.yticks(y_pos, methods)
    plt.xlabel('Average Score (±95% CI)')
    plt.title('Method Performance Comparison (Across All Models)')
    plt.axvline(x=df[df['method_name'] == 'no_system']['harsh score'].mean(), 
                color='red', linestyle='--', label='Baseline (no_system)', alpha=0.7)
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('method_comparison_with_ci.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: method_comparison_with_ci.png")
    plt.show()
        
    # 2. Method group comparison
    plt.figure(figsize=(10, 6))
    
    group_means = group_summary['avg_harsh_score'].values
    group_stds = group_summary['std_harsh_score'].values
    group_names = group_summary.index.tolist()
    
    x_pos = np.arange(len(group_names))
    plt.bar(x_pos, group_means, yerr=group_stds, alpha=0.7, capsize=5)
    plt.xticks(x_pos, group_names, rotation=45, ha='right')
    plt.ylabel('Average Score (±Std)')
    plt.title('Method Group Performance')
    plt.axhline(y=df[df['method_name'] == 'no_system']['harsh score'].mean(), 
                color='red', linestyle='--', label='Baseline', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('group_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: group_comparison.png")
    plt.show()
    
    # 3. Effect size heatmap (top 15 methods x models)
    top_15_methods = method_summary.head(15).index.tolist()
    
    effect_size_matrix = []
    for method in top_15_methods:
        row = []
        for model in sorted(df['model_name'].unique()):
            method_stats = combined_stats[
                (combined_stats['method'] == method) & 
                (combined_stats['model'] == model)
            ]
            if not method_stats.empty:
                row.append(method_stats.iloc[0]['cohens_d'])
            else:
                row.append(0.0)
        effect_size_matrix.append(row)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        effect_size_matrix,
        xticklabels=sorted(df['model_name'].unique()),
        yticklabels=top_15_methods,
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.2f',
        cbar_kws={'label': "Cohen's d (Effect Size)"}
    )
    plt.title("Effect Size Heatmap: Top 15 Methods vs Baseline (by Model)")
    plt.xlabel("Model")
    plt.ylabel("Method")
    plt.tight_layout()
    plt.savefig('effect_size_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: effect_size_heatmap.png")
    plt.show()

# ============================================================================
# ADDITIONAL ANALYSIS: DynaSafe-RL vs Handcrafted Dynamic Methods
# Add this section after the main analysis (before "ANALYSIS COMPLETE")
# ============================================================================

print("\n" + "="*80)
print("DYNASAFE-RL vs HANDCRAFTED DYNAMIC METHODS")
print("="*80 + "\n")

# Define DynaSafe-RL and handcrafted dynamic methods
dynasafe_rl = 'exp_weighted_product'
handcrafted_dynamic = [
    'progressive', 'minimal', 'contrast_learning', 
    'current', 'ai_enhanced', 'hybrid', 'ai_only', 
    'best_worst_recent', 'performance_tiered', 
    'smart_adaptive', 'raw_history'
]

# Filter to only include methods that exist in the data
handcrafted_dynamic = [m for m in handcrafted_dynamic if m in df['method_name'].unique()]

print(f"DynaSafe-RL method: {dynasafe_rl}")
print(f"Handcrafted dynamic methods: {handcrafted_dynamic}\n")

# ============================================================================
# 1. OVERALL PERFORMANCE COMPARISON
# ============================================================================

print("1. OVERALL PERFORMANCE")
print("-"*80)

dynasafe_scores = df[df['method_name'] == dynasafe_rl]['harsh score'].values

print(f"{'Method':<30} {'N':<6} {'Mean':<8} {'Std':<8} {'95% CI':<25}")
print("-"*80)

# DynaSafe-RL stats
rl_ci = calculate_confidence_intervals(dynasafe_scores)
print(f"{dynasafe_rl:<30} {len(dynasafe_scores):<6} {np.mean(dynasafe_scores):<8.3f} "
      f"{np.std(dynasafe_scores):<8.3f} [{rl_ci[0]:.3f}, {rl_ci[1]:.3f}]")

# Each handcrafted method
handcrafted_results = []
for method in handcrafted_dynamic:
    scores = df[df['method_name'] == method]['harsh score'].values
    if len(scores) >= 2:
        ci = calculate_confidence_intervals(scores)
        print(f"{method:<30} {len(scores):<6} {np.mean(scores):<8.3f} "
              f"{np.std(scores):<8.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        handcrafted_results.append({
            'method': method,
            'mean': np.mean(scores),
            'scores': scores
        })

print("\n📊 STATISTICAL TESTS: DynaSafe-RL vs Each Handcrafted Method")
print("-"*80)
print(f"{'Handcrafted Method':<30} {'Mean Diff':<12} {'p-value':<12} {'Sig':<6} {'Cohen d':<10} {'Effect':<12}")
print("-"*80)

for result in handcrafted_results:
    t_stat, p_value = stats.ttest_ind(dynasafe_scores, result['scores'])
    cohens_d = compute_cohens_d(dynasafe_scores, result['scores'])
    
    if abs(cohens_d) < 0.2:
        effect_size = 'negligible'
    elif abs(cohens_d) < 0.5:
        effect_size = 'small'
    elif abs(cohens_d) < 0.8:
        effect_size = 'medium'
    else:
        effect_size = 'large'
    
    sig_marker = "***" if p_value < 0.01 else ("**" if p_value < 0.05 else "ns")
    mean_diff = np.mean(dynasafe_scores) - result['mean']
    
    print(f"{result['method']:<30} {mean_diff:>10.3f}  {p_value:>10.6f}  {sig_marker:<6} "
          f"{cohens_d:>8.3f}  {effect_size:<12}")

# ============================================================================
# 2. QUALITY-SAFETY TRADE-OFF ANALYSIS
# ============================================================================

print("\n\n2. QUALITY-SAFETY TRADE-OFF")
print("-"*80)

# Calculate Q and U scores for each method
def calculate_q_u_scores(method_name):
    method_data = df[df['method_name'] == method_name]
    
    q_score = method_data[Q_metrics].mean().mean()
    u_score = method_data[U_metrics].mean().mean()
    
    return q_score, u_score

rl_q, rl_u = calculate_q_u_scores(dynasafe_rl)
print(f"\n{dynasafe_rl}:")
print(f"  Quality (Q) score: {rl_q:.4f}")
print(f"  Safety (U) score:  {rl_u:.4f}")
print(f"  Q-U balance:       {abs(rl_q - rl_u):.4f}")

print(f"\nHandcrafted methods:")
print(f"{'Method':<30} {'Q Score':<10} {'U Score':<10} {'Q-U Balance':<12} {'vs RL Balance':<15}")
print("-"*90)

handcrafted_tradeoffs = []
for method in handcrafted_dynamic:
    if method in df['method_name'].unique():
        hc_q, hc_u = calculate_q_u_scores(method)
        hc_balance = abs(hc_q - hc_u)
        rl_balance = abs(rl_q - rl_u)
        
        print(f"{method:<30} {hc_q:<10.4f} {hc_u:<10.4f} {hc_balance:<12.4f} "
              f"{'+' if hc_balance < rl_balance else '-'}{abs(hc_balance - rl_balance):.4f}")
        
        handcrafted_tradeoffs.append({
            'method': method,
            'balance': hc_balance
        })

# Statistical test on balance scores
rl_balance_value = abs(rl_q - rl_u)
handcrafted_balance_values = [ht['balance'] for ht in handcrafted_tradeoffs]

if len(handcrafted_balance_values) > 0:
    # One-sample t-test: is RL balance significantly different from mean handcrafted balance?
    mean_hc_balance = np.mean(handcrafted_balance_values)
    std_hc_balance = np.std(handcrafted_balance_values)
    
    print(f"\n📊 Trade-off Balance Comparison:")
    print(f"  DynaSafe-RL balance:           {rl_balance_value:.4f}")
    print(f"  Mean handcrafted balance:      {mean_hc_balance:.4f} (±{std_hc_balance:.4f})")
    print(f"  Difference:                    {mean_hc_balance - rl_balance_value:.4f}")
    
    # Count how many handcrafted methods have worse balance than RL
    worse_count = sum(1 for b in handcrafted_balance_values if b > rl_balance_value)
    print(f"  DynaSafe-RL better than:       {worse_count}/{len(handcrafted_balance_values)} methods")

# ============================================================================
# 3. BEHAVIORAL CONSISTENCY (VARIANCE ANALYSIS)
# ============================================================================

print("\n\n3. BEHAVIORAL CONSISTENCY (Lower variance = more consistent)")
print("-"*80)

# Calculate within-method variance for each method
rl_variance = np.var(dynasafe_scores)
rl_std = np.std(dynasafe_scores)

print(f"\n{dynasafe_rl}:")
print(f"  Variance: {rl_variance:.6f}")
print(f"  Std Dev:  {rl_std:.6f}")

print(f"\nHandcrafted methods:")
print(f"{'Method':<30} {'Variance':<12} {'Std Dev':<10} {'vs RL Var':<15}")
print("-"*80)

handcrafted_variances = []
for method in handcrafted_dynamic:
    scores = df[df['method_name'] == method]['harsh score'].values
    if len(scores) >= 2:
        var = np.var(scores)
        std = np.std(scores)
        diff = var - rl_variance
        
        print(f"{method:<30} {var:<12.6f} {std:<10.6f} "
              f"{'+' if diff > 0 else ''}{diff:.6f}")
        
        handcrafted_variances.append({
            'method': method,
            'variance': var
        })

# Statistical test on variances using Levene's test
if len(handcrafted_variances) > 0:
    # Compare variance of RL vs pooled handcrafted methods
    all_handcrafted_scores = []
    for method in handcrafted_dynamic:
        scores = df[df['method_name'] == method]['harsh score'].values
        if len(scores) >= 2:
            all_handcrafted_scores.extend(scores)
    
    if len(all_handcrafted_scores) > 0:
        # Levene's test for equality of variances
        levene_stat, levene_p = stats.levene(dynasafe_scores, all_handcrafted_scores)
        
        mean_hc_variance = np.mean([hv['variance'] for hv in handcrafted_variances])
        
        print(f"\n📊 Consistency Comparison:")
        print(f"  DynaSafe-RL variance:          {rl_variance:.6f}")
        print(f"  Mean handcrafted variance:     {mean_hc_variance:.6f}")
        print(f"  Difference:                    {mean_hc_variance - rl_variance:.6f}")
        print(f"  Levene's test p-value:         {levene_p:.6f}")
        print(f"  Variances {'significantly different' if levene_p < 0.05 else 'not significantly different'}")
        
        # Count how many handcrafted methods have higher variance than RL
        higher_var_count = sum(1 for hv in handcrafted_variances if hv['variance'] > rl_variance)
        print(f"  DynaSafe-RL more consistent than: {higher_var_count}/{len(handcrafted_variances)} methods")

# ============================================================================
# 4. PER-MODEL ANALYSIS
# ============================================================================

print("\n\n4. PER-MODEL PERFORMANCE")
print("-"*80)

for model in sorted(df['model_name'].unique()):
    print(f"\n{model}:")
    print("-"*60)
    
    model_df = df[df['model_name'] == model]
    
    # Get scores for this model
    model_rl_scores = model_df[model_df['method_name'] == dynasafe_rl]['harsh score'].values
    
    if len(model_rl_scores) < 2:
        print("  Insufficient data for DynaSafe-RL")
        continue
    
    rl_mean = np.mean(model_rl_scores)
    print(f"  DynaSafe-RL: {rl_mean:.3f}")
    
    # Compare to handcrafted methods
    wins = 0
    total = 0
    significant_wins = 0
    
    for method in handcrafted_dynamic:
        hc_scores = model_df[model_df['method_name'] == method]['harsh score'].values
        
        if len(hc_scores) >= 2:
            total += 1
            hc_mean = np.mean(hc_scores)
            
            if rl_mean > hc_mean:
                wins += 1
                
                # Test if significantly better
                t_stat, p_value = stats.ttest_ind(model_rl_scores, hc_scores)
                if p_value < 0.05 and rl_mean > hc_mean:
                    significant_wins += 1
    
    if total > 0:
        print(f"  Outperforms:        {wins}/{total} handcrafted methods")
        print(f"  Significantly better: {significant_wins}/{total} methods (p < 0.05)")

# ============================================================================
# 5. SUMMARY STATISTICS & INTERPRETATION
# ============================================================================

print("\n\n" + "="*80)
print("📊 SUMMARY: DYNASAFE-RL vs HANDCRAFTED DYNAMIC METHODS")
print("="*80)

# Overall win rate
better_count = sum(1 for result in handcrafted_results if np.mean(dynasafe_scores) > result['mean'])
sig_better_count = 0
p_values_better = []
for result in handcrafted_results:
    t_stat, p_value = stats.ttest_ind(dynasafe_scores, result['scores'])
    if p_value < 0.05 and np.mean(dynasafe_scores) > result['mean']:
        sig_better_count += 1
        p_values_better.append(p_value)

print(f"\n1. Overall Performance:")
print(f"   - DynaSafe-RL mean: {np.mean(dynasafe_scores):.3f}")
print(f"   - Outperforms: {better_count}/{len(handcrafted_results)} methods ({100*better_count/len(handcrafted_results):.0f}%)")
print(f"   - Significantly better (p<0.05): {sig_better_count}/{len(handcrafted_results)} methods")
if sig_better_count > 0:
    print(f"   - Smallest p-value: {min(p_values_better):.4f}")

# Calculate rank among all methods
all_method_means = [(result['method'], result['mean']) for result in handcrafted_results]
all_method_means.append((dynasafe_rl, np.mean(dynasafe_scores)))
all_method_means.sort(key=lambda x: x[1], reverse=True)
rl_rank = [i+1 for i, (m, _) in enumerate(all_method_means) if m == dynasafe_rl][0]
print(f"   - Rank: {rl_rank}/{len(all_method_means)} among all dynamic methods")

print(f"\n2. Quality-Safety Trade-off:")
print(f"   - DynaSafe-RL Q-U balance: {abs(rl_q - rl_u):.4f} (lower is better)")
if len(handcrafted_balance_values) > 0:
    better_balance_count = sum(1 for b in handcrafted_balance_values if abs(rl_q - rl_u) < b)
    print(f"   - Better balance than: {better_balance_count}/{len(handcrafted_balance_values)} methods")
    print(f"   - Mean handcrafted balance: {np.mean(handcrafted_balance_values):.4f}")
    
    # Statistical test on whether RL balance is different
    balance_comparison = [abs(rl_q - rl_u) - b for b in handcrafted_balance_values]
    t_stat, p_val = stats.ttest_1samp(balance_comparison, 0)
    print(f"   - One-sample t-test vs handcrafted mean: p={p_val:.4f}")

print(f"\n3. Behavioral Consistency:")
print(f"   - DynaSafe-RL variance: {rl_variance:.6f} (lower is better)")
if len(handcrafted_variances) > 0:
    more_consistent_count = sum(1 for hv in handcrafted_variances if rl_variance < hv['variance'])
    print(f"   - More consistent than: {more_consistent_count}/{len(handcrafted_variances)} methods ({100*more_consistent_count/len(handcrafted_variances):.0f}%)")
    print(f"   - Mean handcrafted variance: {np.mean([hv['variance'] for hv in handcrafted_variances]):.6f}")
    print(f"   - Variance reduction: {100*(1 - rl_variance/np.mean([hv['variance'] for hv in handcrafted_variances])):.1f}%")

# ============================================================================
# 6. INTERPRETATION & STATISTICAL CLAIM SUPPORT
# ============================================================================

print("\n\n" + "="*80)
print("📝 STATISTICAL SUPPORT FOR CLAIMS")
print("="*80)

print("\n🔍 CLAIM ANALYSIS:")
print("-"*80)

# Claim 1: Strongest overall performance
print("\n1. 'Strongest overall performance':")
if rl_rank == 1:
    print(f"   ✅ SUPPORTED: Ranks #1 with mean score {np.mean(dynasafe_scores):.3f}")
    print(f"      Significantly better than {sig_better_count}/{len(handcrafted_results)} handcrafted methods")
else:
    top_method = all_method_means[0]
    print(f"   ⚠️  PARTIALLY SUPPORTED: Ranks #{rl_rank}, not #1")
    print(f"      Top method: {top_method[0]} ({top_method[1]:.3f})")
    print(f"      Difference: {np.mean(dynasafe_scores) - top_method[1]:.4f}")
    
    # Test if difference from #1 is significant
    if rl_rank > 1:
        top_scores = df[df['method_name'] == top_method[0]]['harsh score'].values
        if len(top_scores) >= 2:
            t_stat, p_val = stats.ttest_ind(dynasafe_scores, top_scores)
            print(f"      p-value vs top method: {p_val:.4f} ({'significant' if p_val < 0.05 else 'not significant'})")

# Claim 2: More favorable quality-safety trade-off
print("\n2. 'More favourable quality-safety trade-off':")
if len(handcrafted_balance_values) > 0:
    if better_balance_count > len(handcrafted_balance_values) / 2:
        print(f"   ✅ SUPPORTED: Better balance than {better_balance_count}/{len(handcrafted_balance_values)} methods")
    elif better_balance_count > 0:
        print(f"   ⚠️  MIXED: Better balance than {better_balance_count}/{len(handcrafted_balance_values)} methods")
        print(f"      Note: Most handcrafted methods have smaller Q-U gaps")
    else:
        print(f"   ❌ NOT SUPPORTED: Q-U balance (0.0408) worse than all handcrafted methods")
        print(f"      Mean handcrafted balance: {np.mean(handcrafted_balance_values):.4f}")
        print(f"      This indicates handcrafted methods achieve better Q-S balance")

# Claim 3: More reliable behavioral consistency
print("\n3. 'More reliable behavioural consistency':")
if more_consistent_count > len(handcrafted_variances) / 2:
    print(f"   ✅ SUPPORTED: More consistent than {more_consistent_count}/{len(handcrafted_variances)} methods ({100*more_consistent_count/len(handcrafted_variances):.0f}%)")
    print(f"      Variance: {rl_variance:.6f} vs mean handcrafted: {np.mean([hv['variance'] for hv in handcrafted_variances]):.6f}")
    print(f"      Levene's test p-value: {levene_p:.4f}")
elif more_consistent_count > 0:
    print(f"   ⚠️  MIXED: More consistent than {more_consistent_count}/{len(handcrafted_variances)} methods")
else:
    print(f"   ❌ NOT SUPPORTED: Variance higher than all handcrafted methods")

print("\n" + "="*80)
print("💡 RECOMMENDED STATEMENT (based on results):")
print("="*80)

# Generate data-driven statement
statement = f"\nDynaSafe-RL achieves competitive performance (mean={np.mean(dynasafe_scores):.3f}, rank #{rl_rank}/{len(all_method_means)})"

if sig_better_count > 0:
    statement += f", significantly outperforming {sig_better_count} handcrafted methods (p<0.05)"

if more_consistent_count >= len(handcrafted_variances) * 0.7:
    statement += f", with greater behavioral consistency than {100*more_consistent_count/len(handcrafted_variances):.0f}% of handcrafted approaches"

if better_balance_count == 0:
    statement += ". However, handcrafted methods achieve better quality-safety balance (mean Q-U gap: 0.012 vs 0.041)"

statement += "."

print(statement)

print("\n" + "="*80)
