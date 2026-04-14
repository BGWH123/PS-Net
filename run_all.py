import subprocess
import sys
import os

model_name = "PSNet"

root_path_name = "./data/"
data_name = "custom"

seq_len = 96
pred_lens = [96]
random_seeds = [42]

pa_list = [
 350, 400, 450, 500,
    550, 600, 650, 700, 750, 800, 850, 900
]

def get_drop_cols(target):
    all_targets = ["U", "V", "WS"]
    drop_targets = [t for t in all_targets if t != target]
    fixed_drop = ["lon", "lat", "lev_x", "lev_y", "PHIS", "lev", "source_file"]
    return ",".join(drop_targets + fixed_drop)

def run_once(pred_len, random_seed, pa, target, data_path_name):
    model_id_name = f"wind_pa{pa}_{target}"
    drop_cols = get_drop_cols(target)

    cmd = [
        sys.executable, "-u", "run.py",
        "--is_training", "1",
        "--root_path", root_path_name,
        "--data_path", data_path_name,
        "--model_id", model_id_name,
        "--model", model_name,
        "--data", data_name,
        "--features", "M",
        "--target", target,
        "--seq_len", str(seq_len),
        "--pred_len", str(pred_len),
        "--enc_in", "4",
        "--label_len", "48",
        "--dec_in", "4",
        "--c_out", "4",
        "--cycle", "144",
        "--train_epochs", "5",
        "--patience", "5",
        "--freq" , "h",
        "--dropout", "0.5",
        "--num_workers", "8",
        "--use_gpu", "True",
        "--itr", "1",
        "--batch_size", "64",
        "--learning_rate", "0.001",
        "--random_seed", str(random_seed),
        "--drop_cols", drop_cols,
        "--result_name", f"{model_name}_wind_results_30.txt",
    ]

    print("\nRunning command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    for pa in pa_list:
        for pred_len in pred_lens:
            data_path_name = f"all_wind_speed_{pa}hPa.csv"
            for random_seed in random_seeds:
                run_once(pred_len, random_seed, pa, "U", data_path_name)
                run_once(pred_len, random_seed, pa, "V", data_path_name)
