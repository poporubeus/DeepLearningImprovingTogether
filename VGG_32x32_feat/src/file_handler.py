import os
from ray import tune
from tuning_custom_vgg import train_fn


storage_path = "/home/fv/ray_results"
exp_name = "tune_analyzing_results"



ray_path = os.path.join(storage_path, exp_name)

experiment_path = os.path.join(storage_path, exp_name)
print(f"Loading results from {experiment_path}...")

restored_tuner = tune.Tuner.restore(experiment_path, trainable=train_fn)
result_grid = restored_tuner.get_results()


for i, result in enumerate(result_grid):
    if result.error:
        print(f"Trial #{i} had an error:", result.error)
        continue

    print(
        f"Trial #{i} finished successfully with a mean accuracy metric of:",
        result.metrics["accuracy"]
    )


results_df = result_grid.get_dataframe()
results_df[["training_iteration", "accuracy"]]

file_logs_raytune_csv = results_df.to_csv("/home/fv/storage1/qml/DeepLearningBaby/logs_ray_tune.csv")