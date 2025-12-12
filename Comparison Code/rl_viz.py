import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# 1) Load & preprocess
df = pd.read_csv("generations.csv")
window = 100

# ensure it's sorted so rolling works per function
df = df.sort_values(["function", "timestep"])
grp = df.groupby("function")

# compute only what's needed
df["reward_ma"] = (
    grp["reward_this_step"]
    .rolling(window, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

functions = df["function"].unique().tolist()

# 2) Make a 3×1 subplot
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=[
        "1) Reward per Timestep",
        f"2) {window}-Step Moving Average",
        "3) Reward Distribution",
    ],
    vertical_spacing=0.10
)

# keep track of which trace indices belong to which function
trace_idx = {fn: [] for fn in functions}

# 3) Add the three panels, one function at a time (only the first fn is visible)
for i, fn in enumerate(functions):
    sub = df[df.function == fn]
    visible = (i == 0)

    # raw reward
    trace_idx[fn].append(len(fig.data))
    fig.add_trace(
        go.Scatter(
            x=sub.timestep,
            y=sub.reward_this_step,
            mode="lines+markers",
            name=fn,
            line=dict(color=px.colors.qualitative.Plotly[i]),
            visible=visible
        ),
        row=1, col=1
    )

    # moving average
    trace_idx[fn].append(len(fig.data))
    fig.add_trace(
        go.Scatter(
            x=sub.timestep,
            y=sub.reward_ma,
            mode="lines",
            name=fn,
            line=dict(color=px.colors.qualitative.Plotly[i]),
            visible=visible
        ),
        row=2, col=1
    )

    # histogram
    trace_idx[fn].append(len(fig.data))
    fig.add_trace(
        go.Histogram(
            x=sub.reward_this_step,
            nbinsx=50,
            name=fn,
            marker=dict(color=px.colors.qualitative.Plotly[i]),
            visible=visible
        ),
        row=3, col=1
    )

# 4) Build dropdown buttons
buttons = []
n_tr = len(fig.data)
max_fn_tr = max(max(idxs) for idxs in trace_idx.values())

for fn in functions:
    # start with all False
    vis = [False] * n_tr
    # turn on this function's three traces
    for idx in trace_idx[fn]:
        vis[idx] = True
    # leave any traces beyond those (none here) True—but just in case:
    for idx in range(max_fn_tr + 1, n_tr):
        vis[idx] = True

    buttons.append({
        "label": str(fn),
        "method": "update",
        "args": [
            {"visible": vis},
            {"title": f"Reward Analysis — Function: {fn}"}
        ]
    })

fig.update_layout(
    updatemenus=[dict(
        buttons=buttons,
        x=0, y=1.15,
        xanchor="left", yanchor="top"
    )],
    title=f"Reward Analysis — Function: {functions[0]}",
    height=800,
    showlegend=False
)

# 5) Axis labels
fig.update_xaxes(title_text="Timestep", row=1, col=1)
fig.update_yaxes(title_text="Reward",   row=1, col=1)

fig.update_xaxes(title_text="Timestep",      row=2, col=1)
fig.update_yaxes(title_text="Moving Avg.",   row=2, col=1)

fig.update_xaxes(title_text="Reward Value", row=3, col=1)
fig.update_yaxes(title_text="Count",        row=3, col=1)

# 6) Output
fig.write_html("clean_rewards.html", auto_open=True)
print("Generated clean_rewards.html ")
