import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr

sns.set(style='whitegrid', font_scale=1.0)

# ============================================================================
# DIAGNOSTIC ANALYSIS: WHY DID RL UNDERPERFORM VS DYNAMIC?
# ============================================================================

def categorize_method(method_name):
    """Categorize methods into groups."""
    method = method_name.lower()
    
    rl_methods = [
        'poly', 'exp_weighted_product', 'sqrt_weighted', 'adaptive_softmax', 
        'bayesian_balance', 'exp_weighted_diff', 'sigmoid_diff', 'softmax_weighted', 
        'logistic', 'harmonic_ratio', 'custom_composite','ratio'
    ]
    if any(rl_method in method for rl_method in rl_methods):
        return 'RL'
    
    dynamic_methods = [
        'progressive', 'trajectory_learning', 'minimal', 'contrast_learning', 
        'current', 'ai_enhanced', 'hybrid', 'ai_only', 'best_worst_recent', 
        'performance_tiered', 'smart_adaptive','raw_history'
    ]
    if any(dynamic_method in method for dynamic_method in dynamic_methods):
        return 'Dynamic'
    
    static_methods = [
        'risk_aware', 'chain_of_thought', 'self_consistency', 'few_shot',
        'perspective_taking', 'enhanced_chain_of_thought', 'basic_injection', 
        'enhanced_self_correction', 'improved_few_shot', 'value_reinforcement',
        'self_correction', 'roleplay'
    ]
    if any(static_method in method for static_method in static_methods):
        return 'Static'
    
    if method in ['none', 'no_system']:
        return 'Baseline'
    
    return 'Other'

print("="*80)
print("DIAGNOSTIC ANALYSIS: RL vs DYNAMIC UNDERPERFORMANCE")
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
        'trajectory_learning', 'minimal', 'contrast_learning', 
        'current', 'ai_enhanced', 'hybrid', 'ai_only', 'best_worst_recent', 
        'performance_tiered', 'smart_adaptive',
    ]


df = df[~df['method_name'].isin(methods_to_exclude)]
# Add method categories
df['method_group'] = df['method_name'].apply(categorize_method)



# FILTER OUT 'Other' category from the start
df = df[df['method_group'] != 'Other'].copy()

# Define metrics
Q_metrics = ["AnswerRelevancyMetric", "BiasMetric", "CoherenceMetric"]
U_metrics = [
    "ChildSexualExploitationMetric", "DefamationMetric", "HateMetric", 
    "IllegalContentMetric", "IndiscriminateWeaponsMetric", "IntellectualPropertyMetric",
    "NonViolentCrimesMetric", "PrivacyMetric", "RudeContentMetric", 
    "SexRelatedCrimesMetric", "SexualContentMetric", "SpecializedAdviceMetric",
    "SuicideSelfHarmMetric", "ToxicityMetric", "ViolentCrimesMetric"
]

all_metrics = Q_metrics + U_metrics
df = df.dropna(subset=all_metrics + ["harsh score"]).reset_index(drop=True)

# ============================================================================
# HYPOTHESIS 1: VARIANCE ANALYSIS
# ============================================================================

print("HYPOTHESIS 1: RL METHODS HAVE HIGHER VARIANCE (LESS STABLE)")
print("-"*80)

variance_by_group = df.groupby(['method_group', 'method_name'])['harsh score'].agg([
    'mean', 'std', 'var', 'count'
]).reset_index()

rl_variance = variance_by_group[variance_by_group['method_group'] == 'RL']
dynamic_variance = variance_by_group[variance_by_group['method_group'] == 'Dynamic']

print(f"\nRL Methods:")
print(f"  Average Variance: {rl_variance['var'].mean():.4f}")
print(f"  Average Std: {rl_variance['std'].mean():.4f}")
print(f"  Variance Range: {rl_variance['var'].min():.4f} - {rl_variance['var'].max():.4f}")

print(f"\nDynamic Methods:")
print(f"  Average Variance: {dynamic_variance['var'].mean():.4f}")
print(f"  Average Std: {dynamic_variance['std'].mean():.4f}")
print(f"  Variance Range: {dynamic_variance['var'].min():.4f} - {dynamic_variance['var'].max():.4f}")

# Statistical test
if len(rl_variance) > 1 and len(dynamic_variance) > 1:
    t_stat, p_value = stats.ttest_ind(rl_variance['var'], dynamic_variance['var'])
    print(f"\nLevene's Test for Variance Equality:")
    levene_stat, levene_p = stats.levene(
        df[df['method_group'] == 'RL']['harsh score'],
        df[df['method_group'] == 'Dynamic']['harsh score']
    )
    print(f"  Statistic: {levene_stat:.4f}")
    print(f"  p-value: {levene_p:.4f}")
    if levene_p < 0.05:
        print(f"  ✓ SIGNIFICANT: RL has different variance than Dynamic")
    else:
        print(f"  ✗ NOT SIGNIFICANT: Variance is similar")

# Store for later use
rl_df_temp = df[df['method_group'] == 'RL']
dynamic_df_temp = df[df['method_group'] == 'Dynamic']

# ============================================================================
# HYPOTHESIS 2: CEILING EFFECTS (RL METHODS HIT UPPER BOUND)
# ============================================================================

print("\n" + "="*80)
print("HYPOTHESIS 2: CEILING EFFECTS (Methods hitting upper bound)")
print("-"*80)

# Check how many scores are near 1.0
rl_df_temp = df[df['method_group'] == 'RL']
dynamic_df_temp = df[df['method_group'] == 'Dynamic']

rl_ceiling = (rl_df_temp['harsh score'] > 0.95).sum() / len(rl_df_temp) * 100
dynamic_ceiling = (dynamic_df_temp['harsh score'] > 0.95).sum() / len(dynamic_df_temp) * 100

rl_perfect = (rl_df_temp['harsh score'] == 1.0).sum() / len(rl_df_temp) * 100
dynamic_perfect = (dynamic_df_temp['harsh score'] == 1.0).sum() / len(dynamic_df_temp) * 100

print(f"\nRL Methods:")
print(f"  % scores > 0.95: {rl_ceiling:.2f}%")
print(f"  % perfect scores (1.0): {rl_perfect:.2f}%")
print(f"  Mean score: {rl_df_temp['harsh score'].mean():.3f}")

print(f"\nDynamic Methods:")
print(f"  % scores > 0.95: {dynamic_ceiling:.2f}%")
print(f"  % perfect scores (1.0): {dynamic_perfect:.2f}%")
print(f"  Mean score: {dynamic_df_temp['harsh score'].mean():.3f}")

if dynamic_ceiling > rl_ceiling:
    print(f"\n✓ Dynamic methods hit ceiling MORE often ({dynamic_ceiling:.1f}% vs {rl_ceiling:.1f}%)")
else:
    print(f"\n✗ No ceiling effect detected")

# ============================================================================
# HYPOTHESIS 3: MODEL-SPECIFIC EFFECTS
# ============================================================================

print("\n" + "="*80)
print("HYPOTHESIS 3: MODEL-SPECIFIC PERFORMANCE")
print("-"*80)

model_group_performance = df.groupby(['model_name', 'method_group'])['harsh score'].agg([
    'mean', 'std', 'count'
]).reset_index()

print("\nPerformance by Model and Group:")
print(f"{'Model':<40} {'RL Mean':<10} {'Dynamic Mean':<15} {'Difference':<12}")
print("-"*80)

for model in df['model_name'].unique():
    model_data = model_group_performance[model_group_performance['model_name'] == model]
    rl_mean = model_data[model_data['method_group'] == 'RL']['mean'].values
    dynamic_mean = model_data[model_data['method_group'] == 'Dynamic']['mean'].values
    
    if len(rl_mean) > 0 and len(dynamic_mean) > 0:
        diff = dynamic_mean[0] - rl_mean[0]
        winner = "Dynamic" if diff > 0 else "RL"
        print(f"{model:<40} {rl_mean[0]:>8.3f}  {dynamic_mean[0]:>13.3f}  {diff:>10.3f} ({winner})")

# ============================================================================
# HYPOTHESIS 4: METRIC-SPECIFIC PERFORMANCE (Q vs U)
# ============================================================================

print("\n" + "="*80)
print("HYPOTHESIS 4: Q (QUALITY) vs U (SAFETY) METRIC PERFORMANCE")
print("-"*80)

# Calculate Q and U scores
df['Q_score'] = df[Q_metrics].mean(axis=1)
df['U_score'] = df[U_metrics].mean(axis=1)

rl_q = df[df['method_group'] == 'RL']['Q_score'].mean()
rl_u = df[df['method_group'] == 'RL']['U_score'].mean()
dynamic_q = df[df['method_group'] == 'Dynamic']['Q_score'].mean()
dynamic_u = df[df['method_group'] == 'Dynamic']['U_score'].mean()

print(f"\nRL Methods:")
print(f"  Quality (Q) Score: {rl_q:.3f}")
print(f"  Safety (U) Score:  {rl_u:.3f}")
print(f"  Q/U Ratio: {rl_q/rl_u:.3f}")

print(f"\nDynamic Methods:")
print(f"  Quality (Q) Score: {dynamic_q:.3f}")
print(f"  Safety (U) Score:  {dynamic_u:.3f}")
print(f"  Q/U Ratio: {dynamic_q/dynamic_u:.3f}")

print(f"\nDifferences:")
print(f"  RL - Dynamic (Q): {rl_q - dynamic_q:+.3f}")
print(f"  RL - Dynamic (U): {rl_u - dynamic_u:+.3f}")

if abs(rl_u - dynamic_u) > abs(rl_q - dynamic_q):
    print(f"\n✓ SAFETY (U) metrics show larger difference")
    print(f"  RL methods may be {('worse' if rl_u < dynamic_u else 'better')} at safety")
else:
    print(f"\n✓ QUALITY (Q) metrics show larger difference")

# Statistical tests
t_stat_q, p_q = stats.ttest_ind(
    df[df['method_group'] == 'RL']['Q_score'],
    df[df['method_group'] == 'Dynamic']['Q_score']
)
t_stat_u, p_u = stats.ttest_ind(
    df[df['method_group'] == 'RL']['U_score'],
    df[df['method_group'] == 'Dynamic']['U_score']
)

print(f"\nStatistical Significance:")
print(f"  Q metrics: p = {p_q:.4f} {'***' if p_q < 0.001 else '**' if p_q < 0.05 else 'ns'}")
print(f"  U metrics: p = {p_u:.4f} {'***' if p_u < 0.001 else '**' if p_u < 0.05 else 'ns'}")

# ============================================================================
# HYPOTHESIS 5: SAMPLE SIZE EFFECTS
# ============================================================================

print("\n" + "="*80)
print("HYPOTHESIS 5: SAMPLE SIZE / DATA AVAILABILITY")
print("-"*80)

sample_sizes = df.groupby(['method_group', 'method_name']).size().reset_index(name='n_samples')

rl_samples = sample_sizes[sample_sizes['method_group'] == 'RL']
dynamic_samples = sample_sizes[sample_sizes['method_group'] == 'Dynamic']

print(f"\nRL Methods:")
print(f"  Average samples per method: {rl_samples['n_samples'].mean():.1f}")
print(f"  Min samples: {rl_samples['n_samples'].min()}")
print(f"  Max samples: {rl_samples['n_samples'].max()}")

print(f"\nDynamic Methods:")
print(f"  Average samples per method: {dynamic_samples['n_samples'].mean():.1f}")
print(f"  Min samples: {dynamic_samples['n_samples'].min()}")
print(f"  Max samples: {dynamic_samples['n_samples'].max()}")

# ============================================================================
# HYPOTHESIS 6: CORRELATION WITH INDIVIDUAL METRICS
# ============================================================================

print("\n" + "="*80)
print("HYPOTHESIS 6: METRIC-LEVEL CORRELATION ANALYSIS")
print("-"*80)

print("\nTop metrics correlating with RL success:")
rl_only = df[df['method_group'] == 'RL']
rl_correlations = []
for metric in all_metrics:
    if metric in rl_only.columns:
        corr, p_val = pearsonr(rl_only[metric], rl_only['harsh score'])
        rl_correlations.append({'metric': metric, 'corr': corr, 'p_value': p_val})

rl_corr_df = pd.DataFrame(rl_correlations).sort_values('corr', ascending=False)
print(rl_corr_df.head(10).to_string(index=False))

print("\n\nTop metrics correlating with Dynamic success:")
dynamic_only = df[df['method_group'] == 'Dynamic']
dynamic_correlations = []
for metric in all_metrics:
    if metric in dynamic_only.columns:
        corr, p_val = pearsonr(dynamic_only[metric], dynamic_only['harsh score'])
        dynamic_correlations.append({'metric': metric, 'corr': corr, 'p_value': p_val})

dynamic_corr_df = pd.DataFrame(dynamic_correlations).sort_values('corr', ascending=False)
print(dynamic_corr_df.head(10).to_string(index=False))

# ============================================================================
# HYPOTHESIS 7: OUTLIER ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("HYPOTHESIS 7: OUTLIER / FAILURE MODE ANALYSIS")
print("-"*80)

# Re-filter after Q and U scores are added
rl_df = df[df['method_group'] == 'RL'].copy()
dynamic_df = df[df['method_group'] == 'Dynamic'].copy()

# Find worst performing instances
rl_worst = rl_df.nsmallest(20, 'harsh score')[['method_name', 'model_name', 'harsh score', 'Q_score', 'U_score']]
dynamic_worst = dynamic_df.nsmallest(20, 'harsh score')[['method_name', 'model_name', 'harsh score', 'Q_score', 'U_score']]

print("\nWorst 10 RL instances:")
print(rl_worst.head(10).to_string(index=False))

print("\n\nWorst 10 Dynamic instances:")
print(dynamic_worst.head(10).to_string(index=False))

# Failure rate (score < 0.5)
rl_failure_rate = (rl_df['harsh score'] < 0.5).sum() / len(rl_df) * 100
dynamic_failure_rate = (dynamic_df['harsh score'] < 0.5).sum() / len(dynamic_df) * 100

print(f"\n\nFailure Rates (score < 0.5):")
print(f"  RL: {rl_failure_rate:.2f}%")
print(f"  Dynamic: {dynamic_failure_rate:.2f}%")

if rl_failure_rate > dynamic_failure_rate:
    print(f"\n✓ RL has MORE failures ({rl_failure_rate:.1f}% vs {dynamic_failure_rate:.1f}%)")
    print(f"  This could explain lower average!")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("GENERATING DIAGNOSTIC VISUALIZATIONS")
print("="*80)

# 1. Distribution comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(rl_df['harsh score'], bins=30, alpha=0.7, label='RL', edgecolor='black')
axes[0].hist(dynamic_df['harsh score'], bins=30, alpha=0.7, label='Dynamic', edgecolor='black')
axes[0].axvline(rl_df['harsh score'].mean(), color='red', linestyle='--', label='RL Mean')
axes[0].axvline(dynamic_df['harsh score'].mean(), color='blue', linestyle='--', label='Dynamic Mean')
axes[0].set_xlabel('Harsh Score')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Scores: RL vs Dynamic')
axes[0].legend()

# 2. Box plot comparison
axes[1].boxplot([rl_df['harsh score'], dynamic_df['harsh score']], 
                labels=['RL', 'Dynamic'])
axes[1].set_ylabel('Score')
axes[1].set_title('Score Distribution: RL vs Dynamic')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('rl_vs_dynamic_distributions.png', dpi=300, bbox_inches='tight')
print("✅ Saved: rl_vs_dynamic_distributions.png")
plt.show()

# 3. Q vs U scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(rl_df['Q_score'], rl_df['U_score'], alpha=0.5, label='RL', s=50)
plt.scatter(dynamic_df['Q_score'], dynamic_df['U_score'], alpha=0.5, label='Dynamic', s=50)
plt.xlabel('Quality (Q) Score')
plt.ylabel('Safety (U) Score')
plt.title('Quality vs Safety: RL vs Dynamic Methods')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('q_vs_u_scatter.png', dpi=300, bbox_inches='tight')
print("✅ Saved: q_vs_u_scatter.png")
plt.show()

# 4. Model-specific performance
fig, ax = plt.subplots(figsize=(12, 6))
model_group_pivot = df.groupby(['model_name', 'method_group'])['harsh score'].mean().unstack()
# Only keep RL, Dynamic, and Static columns
cols_to_keep = [col for col in ['RL', 'Dynamic', 'Static'] if col in model_group_pivot.columns]
model_group_pivot = model_group_pivot[cols_to_keep]
model_group_pivot.plot(kind='bar', ax=ax)
ax.set_xlabel('Model')
ax.set_ylabel('Average Score')
ax.set_title('Performance by Model and Method Group')
ax.legend(title='Method Group')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('model_specific_performance.png', dpi=300, bbox_inches='tight')
print("✅ Saved: model_specific_performance.png")
plt.show()

# 5. Variance comparison
fig, ax = plt.subplots(figsize=(10, 6))
variance_comparison = variance_by_group.groupby('method_group')['std'].mean()
# Only keep RL, Dynamic, and Static
variance_comparison = variance_comparison[variance_comparison.index.isin(['RL', 'Dynamic', 'Static'])]
ax.bar(variance_comparison.index, variance_comparison.values)
ax.set_ylabel('Average Standard Deviation')
ax.set_xlabel('Method Group')
ax.set_title('Average Std Dev by Method Group')
plt.tight_layout()
plt.savefig('variance_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: variance_comparison.png")
plt.show()

# ============================================================================
# SUMMARY AND RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY: WHY DID RL UNDERPERFORM?")
print("="*80 + "\n")

findings = []

# Check each hypothesis
if rl_variance['var'].mean() > dynamic_variance['var'].mean() * 1.2:
    findings.append(("HIGH VARIANCE", f"RL methods have {rl_variance['var'].mean()/dynamic_variance['var'].mean():.1f}x higher variance → less stable"))

if rl_failure_rate > dynamic_failure_rate * 1.5:
    findings.append(("MORE FAILURES", f"RL has {rl_failure_rate/dynamic_failure_rate:.1f}x more failure cases (score < 0.5)"))

if abs(rl_u - dynamic_u) > 0.05:
    findings.append(("SAFETY DIFFERENCE", f"RL {'worse' if rl_u < dynamic_u else 'better'} at safety by {abs(rl_u - dynamic_u):.3f}"))

if abs(rl_q - dynamic_q) > 0.05:
    findings.append(("QUALITY DIFFERENCE", f"RL {'worse' if rl_q < dynamic_q else 'better'} at quality by {abs(rl_q - dynamic_q):.3f}"))

# Print findings
for i, (title, description) in enumerate(findings, 1):
    print(f"{i}. {title}")
    print(f"   {description}\n")

if not findings:
    print("No major differences detected. Performance gap may be due to:")
    print("  - Random variation")
    print("  - Specific model architectures")
    print("  - Dataset characteristics")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print("\n1. Consider HYBRID approaches combining RL and Dynamic")
print("2. Investigate worst-case scenarios for RL methods")
print("3. Fine-tune RL reward functions for your specific use case")
print("4. Test with more diverse models to confirm generalizability")
print("\n" + "="*80)
