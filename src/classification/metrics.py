from sklearn.metrics import make_scorer, matthews_corrcoef, accuracy_score, f1_score, precision_score, recall_score


def get_tp(y_true, y_pred):
    if len(y_true) == len(y_pred):
        return sum(couple[0] == couple[1] == 1 for couple in zip(y_true, y_pred))
    else:
        raise ValueError


def get_tn(y_true, y_pred):
    if len(y_true) == len(y_pred):
        return sum(couple[0] == couple[1] == 0 for couple in zip(y_true, y_pred))
    else:
        raise ValueError


def get_fp(y_true, y_pred):
    if len(y_true) == len(y_pred):
        return sum(couple[0] == 0 and couple[1] == 1 for couple in zip(y_true, y_pred))
    else:
        raise ValueError


def get_fn(y_true, y_pred):
    if len(y_true) == len(y_pred):
        return sum(couple[0] == 1 and couple[1] == 0 for couple in zip(y_true, y_pred))
    else:
        raise ValueError


def precision_yes(y_true, y_pred):
    try:
        tp = get_tp(y_true, y_pred)
        fp = get_fp(y_true, y_pred)
        return tp / (tp + fp)
    except ZeroDivisionError:
        return 0


def recall_yes(y_true, y_pred):
    tp = get_tp(y_true, y_pred)
    fn = get_fn(y_true, y_pred)
    return tp / (tp + fn)


def precision_no(y_true, y_pred):
    tn = get_tn(y_true, y_pred)
    fn = get_fn(y_true, y_pred)
    return tn / (tn + fn)


def recall_no(y_true, y_pred):
    tn = get_tn(y_true, y_pred)
    fp = get_fp(y_true, y_pred)
    return tn / (tn + fp)


def f1_no(y_true, y_pred):
    p = precision_no(y_true, y_pred)
    r = recall_no(y_true, y_pred)
    return 2 * ((p * r) / (p + r))


def f1_yes(y_true, y_pred):
    try:
        p = precision_yes(y_true, y_pred)
        r = recall_yes(y_true, y_pred)
        return 2 * ((p * r) / (p + r))
    except ZeroDivisionError:
        return 0


scorers = {'precision_no': make_scorer(score_func=precision_no),
           'recall_no': make_scorer(score_func=recall_no),
           'f1_no': make_scorer(score_func=f1_no),
           'precision_yes': make_scorer(score_func=precision_yes),
           'recall_yes': make_scorer(score_func=recall_yes),
           'f1_yes': make_scorer(score_func=f1_yes),
           'accuracy': 'accuracy',
           'precision_macro': 'precision_macro',
           'recall_macro': 'recall_macro',
           'f1_macro': 'f1_macro',
           'matthews_corrcoef': make_scorer(score_func=matthews_corrcoef),
           }


def compute_metrics(y_true, y_pred):
    return {
        'precision_no': precision_no(y_true, y_pred),
        'recall_no': recall_no(y_true, y_pred),
        'f1_no': f1_no(y_true, y_pred),
        'precision_yes': precision_yes(y_true, y_pred),
        'recall_yes': recall_yes(y_true, y_pred),
        'f1_yes': f1_yes(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
    }
