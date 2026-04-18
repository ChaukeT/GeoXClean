"""
Pseudoflow-based pit optimization wrapper.
Implements maximum closure using min-cut / max-flow for optimal open pit design.
"""

import numpy as np


def pseudoflow_pit_optimization(df, arcs):
    """
    Run pit optimization using pseudoflow or fallback to networkx max-flow.

    Parameters
    ----------
    df : pd.DataFrame
        Block model with 'id' and 'economic_value' columns.
    arcs : list of tuples
        Precedence arcs as (parent_id, child_id).

    Returns
    -------
    pd.DataFrame
        Input dataframe with added 'in_pit' column (1 = in pit, 0 = not).
    """
    n = len(df)
    values = df['economic_value'].values

    # Try pseudoflow library first
    try:
        import pseudoflow as pf
        import networkx as nx

        G = nx.DiGraph()
        source = n
        sink = n + 1

        for i in range(n):
            if values[i] > 0:
                G.add_edge(source, i, capacity=int(abs(values[i]) * 100 + 1))
            if values[i] < 0:
                G.add_edge(i, sink, capacity=int(abs(values[i]) * 100 + 1))

        INF = int(1e12)
        for parent, child in arcs:
            G.add_edge(child, parent, capacity=INF)

        breakpoints, cuts, info = pf.hpf(G, source, sink, const_cap="capacity")

        in_pit = np.zeros(n, dtype=int)
        for node, side in cuts[0].items():
            if isinstance(node, int) and node < n and side == 0:
                in_pit[node] = 1

        df_result = df.copy()
        df_result['in_pit'] = in_pit
        return df_result

    except ImportError:
        pass

    # Fallback to NetworkX max-flow
    import networkx as nx

    print("  Using NetworkX max-flow (Boykov-Kolmogorov)...")

    G = nx.DiGraph()
    source = 's'
    sink = 't'

    for i in range(n):
        if values[i] > 0:
            G.add_edge(source, i, capacity=abs(values[i]))
        elif values[i] < 0:
            G.add_edge(i, sink, capacity=abs(values[i]))

    INF = 1e15
    for parent, child in arcs:
        if G.has_edge(child, parent):
            G[child][parent]['capacity'] += INF
        else:
            G.add_edge(child, parent, capacity=INF)

    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("  Computing min-cut...")

    cut_value, partition = nx.minimum_cut(G, source, sink)
    reachable, non_reachable = partition

    in_pit = np.zeros(n, dtype=int)
    for node in reachable:
        if isinstance(node, int) and node < n:
            in_pit[node] = 1

    df_result = df.copy()
    df_result['in_pit'] = in_pit

    pit_value = df_result.loc[df_result['in_pit'] == 1, 'economic_value'].sum()
    print(f"  Optimal pit value: ${pit_value:,.0f}")

    return df_result
