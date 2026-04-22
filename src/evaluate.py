from sklearn.metrics import log_loss

def evaluate(model, X_val, y_val):
    pred = model.predict_proba(X_val)[:, 1]
    loss = log_loss(y_val, pred)

    print(f"Log Loss: {loss:.4f}")
    return loss