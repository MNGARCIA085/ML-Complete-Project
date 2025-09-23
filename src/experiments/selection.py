

# get best model according to a criterion
def get_best_model(results, recall_threshold=0.8, metric_priority="f1_score"):
    """
    Select the best model based on recall threshold and a priority metric.
    
    Args:
        all_results (list of dict): Each dict should contain "metrics", "hp", "name", "model".
        threshold (float): Minimum recall to be considered a candidate.
        metric_priority (str): Metric to use when multiple candidates meet threshold.

    Returns:
        dict: Best model result dictionary.
    """
    candidates = [res for res in results if res["metrics"]["recall"] >= recall_threshold]
    if candidates:
        return max(candidates, key=lambda r: r["metrics"][metric_priority])
    return max(all_results, key=lambda r: r["metrics"]["recall"])




# gest best model in a single accross al runs for a single experiment
def get_best_model_runs(runs, recall_threshold=0.8):
    candidates = [r for r in runs if r.data.metrics.get("recall", 0) >= recall_threshold]
    if candidates:
        best_run = max(candidates, key=lambda r: r.data.metrics.get("f1_score", 0))
    else:
        best_run = max(runs, key=lambda r: r.data.metrics.get("recall", 0))
    return best_run