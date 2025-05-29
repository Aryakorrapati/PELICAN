import optuna
storage = "sqlite:////eos/user/a/akorrapa/test_study.db"
study = optuna.create_study(study_name="quicktest", storage=storage, direction="maximize", load_if_exists=True)
def objective(trial):
    return trial.suggest_float("x", 0, 1)
study.optimize(objective, n_trials=2)
