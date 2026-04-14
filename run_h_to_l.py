import subprocess
import sys
import os
import re

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0"

model_name = "PSNet"

root_path_name = "./data_hight_low"
data_name = "custom"

seq_len = 96
pred_lens = [96]
random_seeds = [2024]

def get_drop_cols(target):
    all_targets = ["U_l", "V_l", "WS_l"]
    drop_targets = [t for t in all_targets if t != target]
    fixed_drop = []
    return ",".join(drop_targets + fixed_drop)


def run_once(pred_len, random_seed, pa_tag, target, data_path_name):
    model_id_name = f"wind_{pa_tag}_{target}"
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
        "--label_len", "48",
        "--target", target,
        "--seq_len", str(seq_len),
        "--pred_len", str(pred_len),
        "--enc_in", "10",
        "--dec_in", "10",
        "--c_out", "10",
        "--cycle", "144",
        "--train_epochs", "5",
        "--patience", "5",
        "--freq", "h",
        "--dropout", "0.5",
        "--num_workers", "8",
        "--use_gpu", "True",
        "--itr", "1",
        "--batch_size", "64",
        "--learning_rate", "0.001",
        "--random_seed", str(random_seed),
        "--drop_cols", drop_cols,
        "--result_name", f"{model_name}_h_to_l_results5.txt"
    ]

    print("\nRunning command:")
    print(" ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    high_pa = [0.1, 0.3, 1, 2, 3, 10, 20, 30, 40, 50]
    low_pa  = [100, 150, 200, 250, 300, 350, 400, 450, 500,
               550, 600, 650, 700, 750, 800, 850, 900]

    for pa_h in high_pa:
        for pa_l in low_pa:
            data_path_name = f"wind_{pa_h}hPa_{pa_l}hPa.csv"
            if pa_h < -1 or (pa_h == -0.1 and pa_l <= -800):
                continue
            else:
                for pred_len in pred_lens:
                    for random_seed in random_seeds:
                        run_once(pred_len, random_seed,
                                 f"h{pa_h}_l{pa_l}", "U_l", data_path_name)
                        run_once(pred_len, random_seed,
                                 f"h{pa_h}_l{pa_l}", "V_l", data_path_name)
                        run_once(pred_len, random_seed,
                                 f"h{pa_h}_l{pa_l}", "WS_l", data_path_name)
