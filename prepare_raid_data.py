"""
Script chuẩn bị và trích xuất dữ liệu mẫu từ Hugging Face dataset 'liamdugan/raid'.
Tải siêu tốc qua direct Parquet shards, cân bằng số lượng Human vs AI,
chia đều cho các họ mô hình (GPT, LLaMA, Mistral) và các miền (abstracts, books, news).
"""

import os
import sys
import json
import random
import argparse
from typing import Dict, List, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ánh xạ shard mặc định cho các domain trong tập RAID (split='train')
DEFAULT_SHARD_MAPPING = {
    "abstracts": [
        "https://huggingface.co/api/datasets/liamdugan/raid/parquet/raid/train/0.parquet",
        "https://huggingface.co/api/datasets/liamdugan/raid/parquet/raid/train/1.parquet"
    ],
    "books": [
        "https://huggingface.co/api/datasets/liamdugan/raid/parquet/raid/train/4.parquet",
        "https://huggingface.co/api/datasets/liamdugan/raid/parquet/raid/train/5.parquet"
    ],
    "news": [
        "https://huggingface.co/api/datasets/liamdugan/raid/parquet/raid/train/8.parquet",
        "https://huggingface.co/api/datasets/liamdugan/raid/parquet/raid/train/9.parquet"
    ]
}


def map_model_family(model_name: str) -> str:
    """Ánh xạ tên mô hình sinh sang nhóm họ (family) đại diện."""
    m = str(model_name).lower()
    if m == "human":
        return "human"
    if "gpt" in m or "chatgpt" in m:
        return "gpt"
    if "llama" in m:
        return "llama"
    if "mistral" in m:
        return "mistral"
    return None


def save_jsonl(data: List[Dict[str, Any]], path: str):
    """Lưu danh sách dict ra file jsonl."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trích xuất và cân bằng tập dữ liệu RAID từ Hugging Face Parquet shards."
    )
    parser.add_argument("--output_dir", type=str, default="original_text",
                        help="Thư mục lưu các file train, val và test (mặc định: 'original_text')")
    parser.add_argument("--prefix", type=str, default="mini_raid_",
                        help="Tiền tố tên file (mặc định: 'mini_raid_')")
    parser.add_argument("--test_dir", type=str, default="",
                        help="Thư mục phụ để lưu bản sao các tập test domain (nếu có)")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Random seed cho tính tái lập (mặc định: 2026)")
    parser.add_argument("--num_train", type=int, default=8000,
                        help="Tổng số mẫu cho tập Train (mặc định: 8000: 4000 Human, 4000 AI)")
    parser.add_argument("--num_valid", type=int, default=1000,
                        help="Tổng số mẫu cho tập Valid (mặc định: 1000: 500 Human, 500 AI)")
    parser.add_argument("--num_test_per_domain", type=int, default=1000,
                        help="Số mẫu test cho mỗi miền domain (mặc định: 1000: 500 Human, 500 AI)")
    parser.add_argument("--min_words", type=int, default=30,
                        help="Số từ tối thiểu của một văn bản (mặc định: 30)")
    parser.add_argument("--domains", nargs="+", default=["abstracts", "books", "news"],
                        help="Danh sách các domain cần lấy (mặc định: abstracts books news)")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("[!] Lỗi: Chưa cài đặt thư viện 'pandas' và 'pyarrow'.")
        print("    Vui lòng chạy: pip install pandas pyarrow fastparquet")
        sys.exit(1)

    random.seed(args.seed)
    print("=" * 70)
    print(" BẮT ĐẦU TRÍCH XUẤT VÀ TẠO BỘ DỮ LIỆU BALANCED RAID")
    print("=" * 70)
    print(f"- Seed ngẫu nhiên      : {args.seed}")
    print(f"- Thư mục output chính : {args.output_dir}")
    print(f"- Thư mục test phụ     : {args.test_dir}")
    print(f"- Số mẫu Train         : {args.num_train} (50% Human, 50% AI)")
    print(f"- Số mẫu Valid         : {args.num_valid} (50% Human, 50% AI)")
    print(f"- Số mẫu Test / domain : {args.num_test_per_domain} (50% Human, 50% AI)")
    print(f"- Các miền lựa chọn    : {args.domains}")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.test_dir:
        os.makedirs(args.test_dir, exist_ok=True)

    domains = args.domains
    num_domains = len(domains)
    families_ai = ["gpt", "llama", "mistral"]

    # Kho chứa mẫu theo domain và họ model
    domain_pools = {d: {"human": [], "gpt": [], "llama": [], "mistral": []} for d in domains}

    # Tính toán chỉ tiêu cần thu thập mỗi nhóm để không tải dư thừa
    # Cần: test + valid/num_domains + train/num_domains
    test_h_per_dom = args.num_test_per_domain // 2
    test_ai_per_family = (args.num_test_per_domain // 2) // len(families_ai)

    val_h_per_dom = (args.num_valid // 2) // num_domains
    val_ai_per_family = (args.num_valid // 2) // (num_domains * len(families_ai))

    train_h_per_dom = (args.num_train // 2) // num_domains
    train_ai_per_family = (args.num_train // 2) // (num_domains * len(families_ai))

    needed_h_per_dom = test_h_per_dom + val_h_per_dom + train_h_per_dom + 100
    needed_ai_per_family = test_ai_per_family + val_ai_per_family + train_ai_per_family + 100

    print(f"[*] Chỉ tiêu tối thiểu mỗi miền: Human ~ {needed_h_per_dom}, mỗi họ AI ~ {needed_ai_per_family}")

    # Tiến hành tải dữ liệu qua Parquet
    for domain in domains:
        print(f"\n[+] Đang tải dữ liệu miền '{domain}'...")
        urls = DEFAULT_SHARD_MAPPING.get(domain, [])
        if not urls:
            print(f"[!] Cảnh báo: Không tìm thấy URL shard cho domain '{domain}'. Bỏ qua!")
            continue

        for shard_idx, url in enumerate(urls):
            print(f"    -> Đang đọc shard {shard_idx + 1}/{len(urls)}: {url.split('/')[-1]} ...")
            try:
                df = pd.read_parquet(url, columns=["generation", "model", "domain", "attack"])
            except Exception as e:
                print(f"    [!] Lỗi khi đọc Parquet ({url}): {e}")
                continue

            # Lọc các mẫu văn bản gốc không bị tấn công (attack == 'none')
            df = df[df["attack"] == "none"]

            for _, row in df.iterrows():
                family = map_model_family(row["model"])
                if family is None:
                    continue

                pool = domain_pools[domain][family]
                max_needed = needed_h_per_dom if family == "human" else needed_ai_per_family
                if len(pool) >= max_needed:
                    continue

                text = str(row["generation"]).strip()
                if len(text.split()) >= args.min_words:
                    pool.append({
                        "text": text,
                        "label": "human" if family == "human" else "machine",
                        "domain": domain,
                        "model": str(row["model"]),
                        "family": family
                    })

            counts = {k: len(v) for k, v in domain_pools[domain].items()}
            print(f"       Số lượng đã gom miền '{domain}': {counts}")

            # Kiểm tra xem domain này đã đủ chỉ tiêu chưa
            is_domain_full = (
                len(domain_pools[domain]["human"]) >= needed_h_per_dom and
                all(len(domain_pools[domain][fam]) >= needed_ai_per_family for fam in families_ai)
            )
            if is_domain_full:
                print(f"    -> Đã đủ chỉ tiêu mẫu cho miền '{domain}'!")
                break

    print("\n" + "=" * 70)
    print(" TIẾN HÀNH PHÂN BỔ MẪU CHO CÁC TẬP TRAIN, VALID, TEST")
    print("=" * 70)

    train_samples = []
    valid_samples = []
    test_domain_samples = {d: [] for d in domains}

    for domain in domains:
        # Shuffle ngẫu nhiên trong từng pool
        for fam in ["human"] + families_ai:
            random.shuffle(domain_pools[domain][fam])

        # 1. Trích xuất Test cho từng domain
        t_h = domain_pools[domain]["human"][:test_h_per_dom]
        domain_pools[domain]["human"] = domain_pools[domain]["human"][test_h_per_dom:]

        t_ai = []
        for i, fam in enumerate(families_ai):
            cnt = test_ai_per_family + (1 if i == 0 and (args.num_test_per_domain // 2) % 3 != 0 else 0)
            t_ai.extend(domain_pools[domain][fam][:cnt])
            domain_pools[domain][fam] = domain_pools[domain][fam][cnt:]

        test_domain_samples[domain] = t_h + t_ai
        random.shuffle(test_domain_samples[domain])

        # 2. Trích xuất Valid
        v_h = domain_pools[domain]["human"][:val_h_per_dom]
        domain_pools[domain]["human"] = domain_pools[domain]["human"][val_h_per_dom:]

        v_ai = []
        for i, fam in enumerate(families_ai):
            cnt = val_ai_per_family + (1 if i == 0 and ((args.num_valid // 2) // num_domains) % 3 != 0 else 0)
            v_ai.extend(domain_pools[domain][fam][:cnt])
            domain_pools[domain][fam] = domain_pools[domain][fam][cnt:]

        valid_samples.extend(v_h + v_ai)

        # 3. Trích xuất Train
        tr_h = domain_pools[domain]["human"][:train_h_per_dom]
        domain_pools[domain]["human"] = domain_pools[domain]["human"][train_h_per_dom:]

        tr_ai = []
        for i, fam in enumerate(families_ai):
            cnt = train_ai_per_family + (1 if i == 0 and ((args.num_train // 2) // num_domains) % 3 != 0 else 0)
            tr_ai.extend(domain_pools[domain][fam][:cnt])
            domain_pools[domain][fam] = domain_pools[domain][fam][cnt:]

        train_samples.extend(tr_h + tr_ai)

    random.shuffle(train_samples)
    random.shuffle(valid_samples)

    # Lưu kết quả
    prefix = args.prefix
    train_path = os.path.join(args.output_dir, f"{prefix}train.json")
    valid_path = os.path.join(args.output_dir, f"{prefix}val.json")

    save_jsonl(train_samples, train_path)
    save_jsonl(valid_samples, valid_path)

    all_test_samples = []
    for domain, samples in test_domain_samples.items():
        domain_test_file = f"{prefix}test_{domain}.json"
        save_jsonl(samples, os.path.join(args.output_dir, domain_test_file))
        if args.test_dir:
            save_jsonl(samples, os.path.join(args.test_dir, domain_test_file))
        all_test_samples.extend(samples)

    # Tập test gộp toàn bộ các miền
    random.shuffle(all_test_samples)
    all_test_file = f"{prefix}test_all.json"
    save_jsonl(all_test_samples, os.path.join(args.output_dir, all_test_file))
    if args.test_dir:
        save_jsonl(all_test_samples, os.path.join(args.test_dir, all_test_file))

    print("\n" + "=" * 70)
    print(" KẾT QUẢ TRÍCH XUẤT VÀ TẠO DATASET THÀNH CÔNG:")
    print("=" * 70)
    print(f"1. Tập Train ({train_path}):")
    print(f"   - Tổng: {len(train_samples)} mẫu | Human: {sum(1 for x in train_samples if x['label']=='human')} | AI: {sum(1 for x in train_samples if x['label']!='human')}")
    print(f"2. Tập Valid ({valid_path}):")
    print(f"   - Tổng: {len(valid_samples)} mẫu | Human: {sum(1 for x in valid_samples if x['label']=='human')} | AI: {sum(1 for x in valid_samples if x['label']!='human')}")
    print("3. Các tập Test Domain:")
    for domain, samples in test_domain_samples.items():
        print(f"   - Miền '{domain}': {len(samples)} mẫu (50% Human / 50% AI) -> {os.path.join(args.output_dir, f'{prefix}test_{domain}.json')}")
    print(f"   - Toàn bộ miền (All): {len(all_test_samples)} mẫu -> {os.path.join(args.output_dir, all_test_file)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
