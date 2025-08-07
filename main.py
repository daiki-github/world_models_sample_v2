# main.py
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import argparse
import time

# 各スクリプトから関数をインポート
from collect_data import run_collection
from preprocess_data import run_preprocessing
from train_vae import run_train_vae
from train_mdn_rnn import run_train_rnn
from train_controller import run_train_controller

def main():
    parser = argparse.ArgumentParser(description="World Models Training Pipeline")
    # --- finetuneをヘルプメッセージから削除 ---
    parser.add_argument('--steps', nargs='+', default=[], 
                        help="実行したいステップを指定します (例: collect preprocess vae rnn controller)")
    parser.add_argument('--all', action='store_true', 
                        help="データ収集からControllerの学習までを順番に実行します")

    args = parser.parse_args()
    steps = args.steps

    if args.all:
        steps = ['collect', 'preprocess', 'vae', 'rnn', 'controller']
    
    if not steps:
        print("実行するステップが指定されていません。--steps または --all を使用してください。")
        parser.print_help()
        return

    total_start_time = time.time()
    print(f"実行する学習パイプライン: {steps}")

    # --- ステップの実行 ---
    if 'collect' in steps:
        print("\n--- [ステップ 1/5] データ収集を開始します ---")
        step_start_time = time.time()
        run_collection()
        print(f"--- データ収集 完了 (所要時間: {time.time() - step_start_time:.2f}秒) ---\n")

    if 'preprocess' in steps:
        print("\n--- [ステップ 2/5] データの前処理を開始します ---")
        step_start_time = time.time()
        run_preprocessing()
        print(f"--- データの前処理 完了 (所要時間: {time.time() - step_start_time:.2f}秒) ---\n")

    if 'vae' in steps:
        print("\n--- [ステップ 3/5] VAEの学習を開始します ---")
        step_start_time = time.time()
        run_train_vae()
        print(f"--- VAEの学習 完了 (所要時間: {time.time() - step_start_time:.2f}秒) ---\n")

    if 'rnn' in steps:
        print("\n--- [ステップ 4/5] MDN-RNNの学習を開始します ---")
        step_start_time = time.time()
        run_train_rnn()
        print(f"--- MDN-RNNの学習 完了 (所要時間: {time.time() - step_start_time:.2f}秒) ---\n")

    if 'controller' in steps:
        print("\n--- [ステップ 5/5] Controllerの学習を開始します ---")
        step_start_time = time.time()
        run_train_controller()
        print(f"--- Controllerの学習 完了 (所要時間: {time.time() - step_start_time:.2f}秒) ---\n")
        
        
    print("==================================================")
    print("すべての指定された処理が完了しました。")
    print(f"総所要時間: {time.time() - total_start_time:.2f}秒")
    print("==================================================")


if __name__ == "__main__":
    main()