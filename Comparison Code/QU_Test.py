import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from adjustText import adjust_text
import matplotlib as mpl
HAVE_ADJUSTTEXT = True

sns.set(style='whitegrid', font_scale=1.0)

def categorize_method(method_name):
    """Categorize methods into groups for better visualization"""
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
    
    # RL Mathematical/Weighted methods
    rl_methods = [
        'poly', 'exp_weighted_product', 'sqrt_weighted', 'adaptive_softmax', 
        'bayesian_balance', 'exp_weighted_diff', 'sigmoid_diff', 'softmax_weighted', 
        'logistic', 'harmonic_ratio', 'custom_composite','ratio'
    ]
    if any(rl_method in method for rl_method in rl_methods):
        return 'RL'
    

    
    # Static/Prompt-based methods
    static_methods = [
        'risk_aware', 'chain_of_thought', 'self_consistency', 'few_shot',
        'perspective_taking', 'enhanced_chain_of_thought', 'basic_injection', 
        'enhanced_self_correction', 'improved_few_shot', 'value_reinforcement',
        'self_correction', 'roleplay'#, 'rude'
    ]
    if any(static_method in method for static_method in static_methods):
        return 'Static'
    
    # Dynamic/Adaptive methods  
    dynamic_methods = [
        'progressive', 'trajectory_learning', 'minimal', 'contrast_learning', 
        'current', 'ai_enhanced', 'hybrid', 'ai_only', 'best_worst_recent', 
        'performance_tiered', 'smart_adaptive','raw_history'
    ]
    if any(dynamic_method in method for dynamic_method in dynamic_methods):
        return 'Dynamic'
    
    # Baseline methods
    if method in ['none', 'no_system']:
        return 'Baseline'
    
    # Catch any remaining methods
    return 'Other'

# -------------------------
# Load dataset
# -------------------------
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
        'self_correction', 'roleplay'
]
df = df[~df['method_name'].isin(methods_to_exclude)]

metrics = [
    "AnswerRelevancyMetric","CoherenceMetric",
    "ToxicityMetric","ViolentCrimesMetric","NonViolentCrimesMetric",
    "ChildSexualExploitationMetric","IndiscriminateWeaponsMetric","HateMetric",
    "SuicideSelfHarmMetric","PrivacyMetric","IllegalContentMetric",
    "BiasMetric","SexRelatedCrimesMetric","DefamationMetric",
    "SpecializedAdviceMetric","IntellectualPropertyMetric",
    "SexualContentMetric","RudeContentMetric"
]
df = df.dropna(subset=metrics + ["harsh score"]).reset_index(drop=True)
X = df[metrics]
y = df['reward']

# -------------------------
# Define Q and U metric groups
# -------------------------
Q_metrics = [
    "AnswerRelevancyMetric", "BiasMetric", "CoherenceMetric"#, "ContextualRelevancyMetric"
]

U_metrics = [
    "ChildSexualExploitationMetric", "DefamationMetric", "HateMetric", 
    "IllegalContentMetric", "IndiscriminateWeaponsMetric", "IntellectualPropertyMetric",
    "NonViolentCrimesMetric", "PrivacyMetric", "RudeContentMetric", 
    "SexRelatedCrimesMetric", "SexualContentMetric", "SpecializedAdviceMetric",
    "SuicideSelfHarmMetric", "ToxicityMetric", "ViolentCrimesMetric"
]

# -------------------------
# Function definitions
# -------------------------
def compute_weighted_effects(df, importance_weights, selected_metrics, baseline_method='no_system'):
    """Compute weighted effects for a given set of metrics and their importance weights"""
    results = []
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        baseline = model_df[model_df['method_name'] == baseline_method][selected_metrics].mean()
        for method in model_df['method_name'].unique():
            if method == baseline_method:
                continue
            method_avg = model_df[model_df['method_name'] == method][selected_metrics].mean()
            metric_diff = method_avg - baseline # So if no_system was 0.5 and another was 0.7 it would be + 0.2
            weighted_effect = metric_diff * importance_weights # Makes sure the difference is relevant
            temp = pd.DataFrame({
                'model_name': model,
                'method_name': method,
                'metric': selected_metrics,
                'metric_value_diff': metric_diff.values,
                'weighted_effect_on_score': weighted_effect.values
            })
            results.append(temp)
    return pd.concat(results, ignore_index=True)

# -------------------------
# Train Random Forest models
# -------------------------
print("="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"Dataset shape: {df.shape}")
print(f"Number of models: {df['model_name'].nunique()}")
print(f"Models: {df['model_name'].unique().tolist()}")
print(f"Number of methods: {df['method_name'].nunique()}")
print(f"Methods: {df['method_name'].unique().tolist()}")
print(f"Harsh score range: {y.min():.3f} to {y.max():.3f}")
print()

# Train models for all metrics, Q metrics, and U metrics
rf = RandomForestRegressor(n_estimators=500, random_state=42)
rf.fit(X, y)
rf_importance = pd.Series(rf.feature_importances_, index=metrics)

X_Q = df[Q_metrics]
X_U = df[U_metrics]

rf_Q = RandomForestRegressor(n_estimators=500, random_state=42)
rf_Q.fit(X_Q, y)
rf_Q_importance = pd.Series(rf_Q.feature_importances_, index=Q_metrics)

rf_U = RandomForestRegressor(n_estimators=500, random_state=42)
rf_U.fit(X_U, y)
rf_U_importance = pd.Series(rf_U.feature_importances_, index=U_metrics)

# -------------------------
# Print feature importance results
# -------------------------
print("="*80)
print("RANDOM FOREST FEATURE IMPORTANCE - ALL METRICS")
print("="*80)
importance_df = rf_importance.sort_values(ascending=False)
for i, (metric, importance) in enumerate(importance_df.items(), 1):
    print(f"{i:2d}. {metric:30s} {importance:.6f}")
print(f"\nAll Metrics Model R² Score: {rf.score(X, y):.4f}")

print("\n" + "="*80)
print("RANDOM FOREST FEATURE IMPORTANCE - Q METRICS")
print("="*80)
print("Q Metrics :")
for metric in Q_metrics:
    print(f"  - {metric}")
print()
importance_Q_df = rf_Q_importance.sort_values(ascending=False)
for i, (metric, importance) in enumerate(importance_Q_df.items(), 1):
    print(f"{i:2d}. {metric:30s} {importance:.6f}")
print(f"\nQ Metrics Model R² Score: {rf_Q.score(X_Q, y):.4f}")

print("\n" + "="*80)
print("RANDOM FOREST FEATURE IMPORTANCE - U METRICS")
print("="*80)
print("U Metrics:")
for metric in U_metrics:
    print(f"  - {metric}")
print()
importance_U_df = rf_U_importance.sort_values(ascending=False)
for i, (metric, importance) in enumerate(importance_U_df.items(), 1):
    print(f"{i:2d}. {metric:30s} {importance:.6f}")
print(f"\nU Metrics Model R² Score: {rf_U.score(X_U, y):.4f}")

print("\n" + "="*80)
print("FEATURE IMPORTANCE COMPARISON")
print("="*80)


# Calculate group-level importance
total_Q_importance = rf_Q_importance.sum()
total_U_importance = rf_U_importance.sum()
print(f"\nGroup-level analysis:")
print(f"Q Metrics total relative importance: {total_Q_importance:.6f}")
print(f"U Metrics total relative importance: {total_U_importance:.6f}")
print(f"Average importance per Q metric: {total_Q_importance/len(Q_metrics):.6f}")
print(f"Average importance per U metric: {total_U_importance/len(U_metrics):.6f}")
print()

# -------------------------
# Compute weighted effects for all three approaches
# -------------------------
weighted_effects_df = compute_weighted_effects(df, rf_importance, metrics)
weighted_effects_Q_df = compute_weighted_effects(df, rf_Q_importance, Q_metrics)
weighted_effects_U_df = compute_weighted_effects(df, rf_U_importance, U_metrics)

# -------------------------
# Q vs U Analysis - Using Importance Scores
# -------------------------
def analyze_Q_vs_U_effects(df, baseline_method='no_system'):
    """Analyze the combined effects of Q vs U metrics on harsh scores"""
    results = []
    
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        
        for method in model_df['method_name'].unique():
            if method == baseline_method:
                continue
                
            method_data = model_df[model_df['method_name'] == method]
            baseline_data = model_df[model_df['method_name'] == baseline_method]
            
            if len(method_data) == 0 or len(baseline_data) == 0:
                continue
            
            # Calculate average scores for Q and U metrics
            method_Q_avg = method_data[Q_metrics].mean().mean()
            baseline_Q_avg = baseline_data[Q_metrics].mean().mean()
            Q_improvement = method_Q_avg - baseline_Q_avg
            
            method_U_avg = method_data[U_metrics].mean().mean()
            baseline_U_avg = baseline_data[U_metrics].mean().mean()
            U_improvement = method_U_avg - baseline_U_avg
            
            # Calculate weighted Q and U effects using importance
            Q_weighted_effect = 0
            U_weighted_effect = 0
            
            for metric in Q_metrics:
                metric_diff = method_data[metric].mean() - baseline_data[metric].mean()
                Q_weighted_effect += metric_diff * rf_Q_importance[metric]
            
            for metric in U_metrics:
                metric_diff = method_data[metric].mean() - baseline_data[metric].mean()
                U_weighted_effect += metric_diff * rf_U_importance[metric]
            
            # Calculate actual harsh score change
            harsh_score_change = method_data['harsh score'].mean() - baseline_data['harsh score'].mean()
            
            harsh_score = method_data['harsh score'].mean()
            
            results.append({
                'model_name': model,
                'method_name': method,
                'Q_raw_improvement': Q_improvement,
                'U_raw_improvement': U_improvement,
                'Q_weighted_effect': Q_weighted_effect,
                'U_weighted_effect': U_weighted_effect,
                'total_predicted_effect': Q_weighted_effect + U_weighted_effect,
                'actual_harsh_score_change': harsh_score_change,
                # Q/Q+U
                'Q_dominance': Q_weighted_effect / (abs(Q_weighted_effect) + abs(U_weighted_effect)) if (abs(Q_weighted_effect) + abs(U_weighted_effect)) > 0 else 0,
                'U_dominance': U_weighted_effect / (abs(Q_weighted_effect) + abs(U_weighted_effect)) if (abs(Q_weighted_effect) + abs(U_weighted_effect)) > 0 else 0,
                'harsh_score': harsh_score
            })
    
    return pd.DataFrame(results)

Q_vs_U_analysis = analyze_Q_vs_U_effects(df)
Q_vs_U_analysis['method_group'] = Q_vs_U_analysis['method_name'].apply(categorize_method)

# -------------------------
# Print detailed results
# -------------------------
print("="*80)
print("DETAILED WEIGHTED EFFECTS FOR ALL METRICS AND MODELS")
print("="*80)

# Print results for each model
for model in sorted(weighted_effects_df['model_name'].unique()):
    print(f"\n{'='*60}")
    print(f"MODEL: {model}")
    print(f"{'='*60}")
    
    model_df = weighted_effects_df[weighted_effects_df['model_name'] == model]
    
    # Print results for each method within this model
    for method in sorted(model_df['method_name'].unique()):
        print(f"\n{'-'*40}")
        print(f"Method: {method}")
        print(f"{'-'*40}")
        
        method_df = model_df[model_df['method_name'] == method].copy()
        method_df = method_df.sort_values('weighted_effect_on_score', key=abs, ascending=False)
        
        print(f"{'Rank':<4} {'Metric':<30} {'Value Diff':<12} {'Weighted Effect':<15} {'Importance':<12}")
        print("-" * 80)
        
        for i, (_, row) in enumerate(method_df.iterrows(), 1):
            metric = row['metric']
            value_diff = row['metric_value_diff']
            weighted_effect = row['weighted_effect_on_score']
            importance = rf_importance[metric]
            
            print(f"{i:<4} {metric:<30} {value_diff:>10.6f}  {weighted_effect:>13.6f}  {importance:>10.6f}")
        
        # Summary statistics for this method
        total_weighted_effect = method_df['weighted_effect_on_score'].sum()
        positive_effects = method_df[method_df['weighted_effect_on_score'] > 0]['weighted_effect_on_score'].sum()
        negative_effects = method_df[method_df['weighted_effect_on_score'] < 0]['weighted_effect_on_score'].sum()
        
        print("-" * 80)
        print(f"Total Weighted Effect: {total_weighted_effect:>12.6f}")
        print(f"Positive Effects Sum:  {positive_effects:>12.6f}")
        print(f"Negative Effects Sum:  {negative_effects:>12.6f}")

# -------------------------
# Summary statistics
# -------------------------
print("\n" + "="*80)
print("Q vs U DOMINANCE ANALYSIS")
print("="*80)
print("Analysis showing whether Quality (Q) or Safety (U) metrics drive harsh score changes")
print()

print("Method Performance Summary:")
print(f"{'Method':<20} {'Q Weighted':<12} {'U Weighted':<12} {'Total Pred':<12} {'Actual':<12} {'Q Dom':<8} {'U Dom':<8}")
print("-" * 95)

for _, row in Q_vs_U_analysis.iterrows():
    print(f"{row['method_name']:<20} {row['Q_weighted_effect']:>10.4f}  {row['U_weighted_effect']:>10.4f}  "
          f"{row['total_predicted_effect']:>10.4f}  {row['actual_harsh_score_change']:>10.4f}  "
          f"{row['Q_dominance']:>6.3f}  {row['U_dominance']:>6.3f}")

# Correlation analysis
from scipy.stats import pearsonr

if len(Q_vs_U_analysis) > 1:
    pred_actual_corr, pred_actual_p = pearsonr(Q_vs_U_analysis['total_predicted_effect'], 
                                               Q_vs_U_analysis['actual_harsh_score_change'])
    
    print(f"\nPredictive Power:")
    print(f"Correlation between predicted and actual harsh score changes: {pred_actual_corr:.4f} (p={pred_actual_p:.4f})")

# Method categorization
print(f"\nMethod Categorization:")
Q_dominant_methods = Q_vs_U_analysis[Q_vs_U_analysis['Q_weighted_effect'].abs() > Q_vs_U_analysis['U_weighted_effect'].abs()]
U_dominant_methods = Q_vs_U_analysis[Q_vs_U_analysis['U_weighted_effect'].abs() > Q_vs_U_analysis['Q_weighted_effect'].abs()]

print(f"Q-Dominant Methods ({len(Q_dominant_methods)}): {Q_dominant_methods['method_name'].tolist()}")
print(f"U-Dominant Methods ({len(U_dominant_methods)}): {U_dominant_methods['method_name'].tolist()}")

if len(Q_dominant_methods) > 0:
    print(f"\nQ-Dominant Methods - Average Effects:")
    print(f"  Average Q weighted effect: {Q_dominant_methods['Q_weighted_effect'].mean():>8.4f}")
    print(f"  Average U weighted effect: {Q_dominant_methods['U_weighted_effect'].mean():>8.4f}")
    print(f"  Average harsh score change: {Q_dominant_methods['actual_harsh_score_change'].mean():>8.4f}")

if len(U_dominant_methods) > 0:
    print(f"\nU-Dominant Methods - Average Effects:")
    print(f"  Average Q weighted effect: {U_dominant_methods['Q_weighted_effect'].mean():>8.4f}")
    print(f"  Average U weighted effect: {U_dominant_methods['U_weighted_effect'].mean():>8.4f}")
    print(f"  Average harsh score change: {U_dominant_methods['actual_harsh_score_change'].mean():>8.4f}")

# -------------------------
# Importance-Based Predictions vs Reality
# -------------------------
print("\n" + "="*80)
print("IMPORTANCE-BASED PREDICTION VALIDATION")
print("="*80)

# Create aggregate Q and U scores for each data point
df['Q_aggregate'] = df[Q_metrics].mean(axis=1)
df['U_aggregate'] = df[U_metrics].mean(axis=1)

# Calculate importance-weighted aggregate scores
df['Q_weighted_score'] = 0
df['U_weighted_score'] = 0

for metric in Q_metrics:
    df['Q_weighted_score'] += df[metric] * rf_Q_importance[metric]
    
for metric in U_metrics:
    df['U_weighted_score'] += df[metric] * rf_U_importance[metric]

# Correlations with harsh score
q_harsh_corr, q_harsh_p = pearsonr(df['Q_weighted_score'], df['harsh score'])
u_harsh_corr, u_harsh_p = pearsonr(df['U_weighted_score'], df['harsh score'])
q_raw_harsh_corr, q_raw_harsh_p = pearsonr(df['Q_aggregate'], df['harsh score'])
u_raw_harsh_corr, u_raw_harsh_p = pearsonr(df['U_aggregate'], df['harsh score'])

print("Correlation with Harsh Score:")
print(f"{'Metric Type':<25} {'Raw Correlation':<16} {'Weighted Correlation':<20}")
print("-" * 65)
print(f"{'Q (Quality) Metrics':<25} {q_raw_harsh_corr:>8.4f} (p={q_raw_harsh_p:.3f})   {q_harsh_corr:>8.4f} (p={q_harsh_p:.3f})")
print(f"{'U (Safety) Metrics':<25} {u_raw_harsh_corr:>8.4f} (p={u_raw_harsh_p:.3f})   {u_harsh_corr:>8.4f} (p={u_harsh_p:.3f})")

improvement_ratio = abs(q_harsh_corr) / abs(q_raw_harsh_corr) if q_raw_harsh_corr != 0 else float('inf')
u_improvement_ratio = abs(u_harsh_corr) / abs(u_raw_harsh_corr) if u_raw_harsh_corr != 0 else float('inf')

print(f"\nImportance Weighting Improvement:")
print(f"Q metrics correlation improved by factor of: {improvement_ratio:.3f}")
print(f"U metrics correlation improved by factor of: {u_improvement_ratio:.3f}")

# Which is more predictive?
if abs(q_harsh_corr) > abs(u_harsh_corr):
    print(f"\n Q (Quality) metrics are more predictive of harsh scores")
    print(f"   Q weighted correlation: {q_harsh_corr:.4f}")
    print(f"   U weighted correlation: {u_harsh_corr:.4f}")
else:
    print(f"\n U (Safety) metrics are more predictive of harsh scores")
    print(f"   U weighted correlation: {u_harsh_corr:.4f}")
    print(f"   Q weighted correlation: {q_harsh_corr:.4f}")

print()
print("="*80)

method_summary = weighted_effects_df.groupby('method_name').agg({
    'weighted_effect_on_score': ['sum', 'mean', 'std', 'count']
}).round(6)
method_summary.columns = ['Total_Effect', 'Mean_Effect', 'Std_Effect', 'Count']
method_summary = method_summary.sort_values('Mean_Effect', ascending=False)

print(f"{'Method':<20} {'Total Effect':<12} {'Mean Effect':<12} {'Std Effect':<12} {'Count':<8}")
print("-" * 70)
for method, row in method_summary.iterrows():
    print(f"{method:<20} {row['Total_Effect']:>10.6f}  {row['Mean_Effect']:>10.6f}  {row['Std_Effect']:>10.6f}  {row['Count']:>6.0f}")

print("\n" + "="*80)
print("SUMMARY BY METHOD - Q METRICS ONLY")
print("="*80)

method_summary_Q = weighted_effects_Q_df.groupby('method_name').agg({
    'weighted_effect_on_score': ['sum', 'mean', 'std', 'count']
}).round(6)
method_summary_Q.columns = ['Total_Effect', 'Mean_Effect', 'Std_Effect', 'Count']
method_summary_Q = method_summary_Q.sort_values('Mean_Effect', ascending=False)

print(f"{'Method':<20} {'Total Effect':<12} {'Mean Effect':<12} {'Std Effect':<12} {'Count':<8}")
print("-" * 70)
for method, row in method_summary_Q.iterrows():
    print(f"{method:<20} {row['Total_Effect']:>10.6f}  {row['Mean_Effect']:>10.6f}  {row['Std_Effect']:>10.6f}  {row['Count']:>6.0f}")

print("\n" + "="*80)
print("SUMMARY BY METHOD - U METRICS ONLY")
print("="*80)

method_summary_U = weighted_effects_U_df.groupby('method_name').agg({
    'weighted_effect_on_score': ['sum', 'mean', 'std', 'count']
}).round(6)
method_summary_U.columns = ['Total_Effect', 'Mean_Effect', 'Std_Effect', 'Count']
method_summary_U = method_summary_U.sort_values('Mean_Effect', ascending=False)

print(f"{'Method':<20} {'Total Effect':<12} {'Mean Effect':<12} {'Std Effect':<12} {'Count':<8}")
print("-" * 70)
for method, row in method_summary_U.iterrows():
    print(f"{method:<20} {row['Total_Effect']:>10.6f}  {row['Mean_Effect']:>10.6f}  {row['Std_Effect']:>10.6f}  {row['Count']:>6.0f}")

print("\n" + "="*80)
print("SUMMARY BY METRIC (ACROSS ALL MODELS AND METHODS)")
print("="*80)

metric_summary = weighted_effects_df.groupby('metric').agg({
    'weighted_effect_on_score': ['sum', 'mean', 'std', 'count'],
    'metric_value_diff': ['mean', 'std']
}).round(6)
metric_summary.columns = ['Total_Weighted', 'Mean_Weighted', 'Std_Weighted', 'Count', 'Mean_Value_Diff', 'Std_Value_Diff']
metric_summary = metric_summary.sort_values('Mean_Weighted', key=abs, ascending=False)

print(f"{'Metric':<30} {'Mean Weighted':<13} {'Mean Value Diff':<15} {'Importance':<12}")
print("-" * 75)
for metric, row in metric_summary.iterrows():
    importance = rf_importance[metric]
    print(f"{metric:<30} {row['Mean_Weighted']:>11.6f}  {row['Mean_Value_Diff']:>13.6f}  {importance:>10.6f}")

# Ensure we have top_weighted_effects_df (top N metrics per model+method)
top_n = 5
if 'top_weighted_effects_df' not in globals():
    top_metrics_list = []
    if 'weighted_effects_df' in globals() and not weighted_effects_df.empty:
        for (m, meth), group in weighted_effects_df.groupby(['model_name', 'method_name']):
            group_sorted = group.loc[group['weighted_effect_on_score'].abs().sort_values(ascending=False).index]
            top_metrics = group_sorted.head(top_n)
            if not top_metrics.empty:
                top_metrics_list.append(top_metrics)
        top_weighted_effects_df = pd.concat(top_metrics_list, ignore_index=True) if top_metrics_list else pd.DataFrame(columns=weighted_effects_df.columns)
    else:
        top_weighted_effects_df = pd.DataFrame(columns=[
            'model_name','method_name','metric','metric_value_diff','weighted_effect_on_score'
        ])

models = Q_vs_U_analysis['model_name'].unique()

for model in models:
    model_qvudf = Q_vs_U_analysis[Q_vs_U_analysis['model_name'] == model].copy()
    if model_qvudf.empty:
        continue
    # Print method group summary for this model
    print(f"\n{model} - Method Group Summary:")
    print("-" * 50)
    group_summary = model_qvudf.groupby('method_group').agg({
        'Q_weighted_effect': ['mean', 'count'],
        'U_weighted_effect': 'mean',
        'total_predicted_effect': 'mean'
    }).round(4)
    
    for group in group_summary.index:
        count = int(group_summary.loc[group, ('Q_weighted_effect', 'count')])
        q_avg = group_summary.loc[group, ('Q_weighted_effect', 'mean')]
        u_avg = group_summary.loc[group, ('U_weighted_effect', 'mean')]
        total_avg = group_summary.loc[group, ('total_predicted_effect', 'mean')]
        print(f"{group:12s} ({count:2d}): Q={q_avg:+.3f}, U={u_avg:+.3f}, Total={total_avg:+.3f}")

    # human-friendly short labels for annotation
    def short_label(meth, max_len=40):
        label = f"{meth}"
        return label if len(label) <= max_len else (label[:max_len-3] + '...')

    model_qvudf['method_label'] = model_qvudf['method_name'].apply(short_label)

    # -------------------------
    # Figure 1 - Clean Scatter: Q vs U (No Labels)
    # -------------------------
    plt.figure(figsize=(8, 6))
    scatter = sns.scatterplot(
        data=model_qvudf,
        x='Q_weighted_effect',
        y='U_weighted_effect',
        hue='harsh_score',
        size='harsh_score',
        sizes=(80, 300),
        palette='viridis',
        alpha=0.9,
        edgecolor='k',
        linewidth=0.4,
        legend=False
    )
    plt.axhline(0, color='gray', linestyle='--', alpha=0.6)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.6)
    plt.title(f"{model} — q vs u")
    plt.xlabel('q (Quality) Weighted Effect')
    plt.ylabel('u (Safety) Weighted Effect')

    # colorbar
    ax = plt.gca()
    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=model_qvudf['harsh_score'].min(),
                                vmax=model_qvudf['harsh_score'].max())
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(model_qvudf['harsh_score'])   
    plt.colorbar(sm, ax=ax, label='Harsh Score', pad=0.02)

    plt.tight_layout()
    plt.show()

    
    model_qvudf = Q_vs_U_analysis[Q_vs_U_analysis['model_name'] == model].copy()
    if model_qvudf.empty:
        continue

    # Add method grouping
    model_qvudf['method_group'] = model_qvudf['method_name'].apply(categorize_method)
    
    # human-friendly short labels for annotation
    def short_label(meth, max_len=40):
        label = f"{meth}"
        return label if len(label) <= max_len else (label[:max_len-3] + '...')

    model_qvudf['method_label'] = model_qvudf['method_name'].apply(short_label)

    # -------------------------
    # Figure 2 - Enhanced Clean Scatter: Q vs U coloured by method group (No Labels)
    # -------------------------
    plt.figure(figsize=(10, 7))
    
    # Define colours for each group
    group_colors = {
        'RL': '#FF6B6B',             # Red
        'Static': '#45B7D1',         # Blue
        'Dynamic': '#96CEB4',        # Green
        'Other': '#FECA57',          # Yellow
        'RL Dynamic': '#9B59B6'      # Purple
    }
    
    # Create scatter plot with method group coloring
    for group in model_qvudf['method_group'].unique():
        group_data = model_qvudf[model_qvudf['method_group'] == group]
        plt.scatter(
            group_data['Q_weighted_effect'],
            group_data['U_weighted_effect'], 
            c=group_colors.get(group, '#CCCCCC'),
            s=group_data['harsh_score'].abs() * 20 + 80,
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5,
            label=f'{group} ({len(group_data)})'
        )
    
    plt.axhline(0, color='gray', linestyle='--', alpha=0.6)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.6)
    plt.title(f"{model} — q vs u Effects by Method Type")
    plt.xlabel('q (Quality) Weighted Effect')
    plt.ylabel('u (Safety) Weighted Effect')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()
    

    # -------------------------
    # Figure 3 - Grouped Horizontal Bars: Q vs U per method
    # -------------------------
    plot_df = model_qvudf.sort_values('total_predicted_effect', ascending=True).reset_index(drop=True)
    methods = plot_df['method_label']
    y_pos = np.arange(len(plot_df))

    plt.figure(figsize=(10, max(4, 0.35 * len(plot_df))))
    bar_q = plt.barh(y_pos - 0.18, plot_df['Q_weighted_effect'], height=0.36, label='q (Quality)', alpha=0.9)
    bar_u = plt.barh(y_pos + 0.18, plot_df['U_weighted_effect'], height=0.36, label='u (Safety)', alpha=0.9)

    plt.yticks(y_pos, methods, fontsize=9)
    plt.gca().invert_yaxis()
    plt.axvline(0, color='black', linewidth=0.6)
    plt.xlabel('Weighted Effect on Harsh Score')
    plt.title(f"{model} — q vs u Weighted Effects by Method ")
    plt.legend()

    # annotate bar values
    def annotate_hbars(bars):
        for bar in bars:
            w = bar.get_width()
            if abs(w) > 1e-5:
                x = bar.get_x() + w
                y = bar.get_y() + bar.get_height() / 2
                ha = 'left' if w >= 0 else 'right'
                offset = 4 if w >= 0 else -4
                plt.gca().annotate(f"{w:.3f}", xy=(x, y), xytext=(offset, 0), textcoords='offset points',
                                   va='center', ha=ha, fontsize=8, color='black')
    annotate_hbars(bar_q)
    annotate_hbars(bar_u)

    plt.tight_layout()
    plt.show()

    # -------------------------
    # Figure 4 - Clean Heatmap: Top 5 GLOBAL weighted metrics per method 
    # -------------------------
    model_top = top_weighted_effects_df[top_weighted_effects_df['model_name'] == model].copy()
    if model_top.empty and 'weighted_effects_df' in globals():
        # compute top_n per method for this model
        per_model_list = []
        mod_df = weighted_effects_df[weighted_effects_df['model_name'] == model]
        for method, g in mod_df.groupby('method_name'):
            g_sorted = g.loc[g['weighted_effect_on_score'].abs().sort_values(ascending=False).index]
            per_model_list.append(g_sorted.head(top_n))
        model_top = pd.concat([t for t in per_model_list if not t.empty], ignore_index=True) if per_model_list else pd.DataFrame()

    if not model_top.empty:
        metric_frequency = model_top['metric'].value_counts().head(5).index.tolist()
        
        global_top_metrics = model_top.groupby('metric')['weighted_effect_on_score'].apply(lambda x: x.abs().mean()).nlargest(5).index.tolist()
        
        # Filter to only these top 5 metrics
        model_top_filtered = model_top[model_top['metric'].isin(global_top_metrics)].copy()
        
        # wrap metric names
        def wrap_label(s, width=18):
            return "\n".join(textwrap.wrap(s, width=width))
        model_top_filtered['metric_wrapped'] = model_top_filtered['metric'].apply(lambda s: wrap_label(s, width=18))

        pivot = model_top_filtered.pivot(index='method_name', columns='metric_wrapped', values='weighted_effect_on_score').fillna(0)
        
        # Calculate figure size
        fig_height = max(6, 0.6 * pivot.shape[0] + 3)
        fig_width = max(10, 0.8 * pivot.shape[1] + 4)  # Will be smaller since only 5 columns max
        
        plt.figure(figsize=(fig_width, fig_height))
        
        ax = sns.heatmap(
            pivot, 
            annot=False,  # Remove numerical annotations
            fmt=".3f", 
            cmap="RdBu", 
            center=0, 
            cbar_kws={'shrink': 0.6}, 
            linewidths=0.5, 
            linecolor='gray'
        )
        
        plt.title(f"{model} — Top 5 Weighted Metrics", pad=20)
        plt.ylabel("Method")
        plt.xlabel("Metric")
        
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        
        plt.subplots_adjust(
            left=0.15,
            bottom=0.25,
            right=0.95,
            top=0.90
        )
        
        plt.show()
        
        print(f"Showing {len(metric_frequency)} metrics: {metric_frequency}")

    else:
        print(f"[Info] No top metrics available for heatmap for model: {model}")
