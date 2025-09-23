


# gest best model in a single accross al runs for a single experiment
def get_best_model(runs, recall_threshold=0.8):
    candidates = [r for r in runs if r.data.metrics.get("recall", 0) >= recall_threshold]
    if candidates:
        best_run = max(candidates, key=lambda r: r.data.metrics.get("f1_score", 0))
    else:
        best_run = max(runs, key=lambda r: r.data.metrics.get("recall", 0))
    return best_run