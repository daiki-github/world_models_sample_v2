import torch

def run_gpu_check():
    print("--- GPU環境の診断を開始します ---")
    print(f"PyTorchのバージョン: {torch.__version__}")
    
    # CUDAが利用可能かチェック
    is_available = torch.cuda.is_available()
    
    if is_available:
        print("\n✅ 成功: GPUは正常に認識されています。")
        # 利用可能なGPUの数を表示
        device_count = torch.cuda.device_count()
        print(f"利用可能なGPUの数: {device_count}")
        
        # 各GPUの情報を表示
        for i in range(device_count):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("\n❌ 失敗: GPUが利用できません。PyTorchがGPUを認識できていません。")
        print("CPUバージョンのPyTorchがインストールされているか、NVIDIAドライバに問題がある可能性があります。")

    print("\n--- 診断を終了します ---")

if __name__ == "__main__":
    run_gpu_check()