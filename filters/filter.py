import pandas as pd

def cusum_filter(g_raw, h):
    """
    CUSUM filter that is used to calculate t_events (points of change detection)
    for Triple-Barrier method.

    :param g_raw: (pd.Series) Time Series data (transformed into various bars)
    :param h: (int) Threshold for the CUSUM filter

    :return: (pd.Series) Returns t_events
    """
    t_events = []
    s_pos = 0
    s_neg = 0
    diff = g_raw.diff()

    for i in diff.index[1:]:
        s_pos, s_neg = max(0, s_pos + diff.loc[i]), min(0, s_neg + diff.loc[i])
        if s_neg <- h:
            s_neg = 0;t_events.append(i)
        elif s_pos > h:
            s_pos = 0;t_events.append(i)

    return pd.Datetimeindex(t_events)



