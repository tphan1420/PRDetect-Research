import os
import sys
import json
import argparse
import glob
from typing import List, Dict, Any, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFAULT_OUTPUT_DIR = "original_text"
DATASETS_DIR = "datasets"

def list_available_datasets():
    """Liệt kê các tập dữ liệu có sẵn trong thư mục datasets/"""
    print("\n" + "=" * 70)
    print(" DANH SÁCH CÁC TẬP DỮ LIỆU CÓ SẴN (AVAILABLE DATASETS)")
    print("=" * 70)
    
    raid_files = glob.glob(os.path.join(DATASETS_DIR, "raid_dataset", "*.json"))
    print(f"\n[+] RAID Dataset ({len(raid_files)} files):")
    for f in raid_files[:10]:
        print(f"    - {f}")
    if len(raid_files) > 10:
        print(f"    ... và {len(raid_files) - 10} file khác trong datasets/raid_dataset/")
        
    detectrl_dirs = glob.glob(os.path.join(DATASETS_DIR, "detectrl_dataset", "*"))
    print(f"\n[+] DetectRL Dataset ({len(detectrl_dirs)} folders):")
    for d in detectrl_dirs:
        if os.path.isdir(d):
            sub_files = glob.glob(os.path.join(d, "*.json"))
            folder_name = os.path.basename(d)
            print(f"    * Thư mục [{folder_name}] ({len(sub_files)} files):")
            for sf in sub_files[:3]:
                print(f"        - {sf}")
            if len(sub_files) > 3:
                print(f"        ... ({len(sub_files) - 3} file khác)")
    print("=" * 70 + "\n")

def inspect_dataset_fields(file_path: str):
    """Kiểm tra các trường văn bản có sẵn trong file JSON"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            sample = data[0]
            print(f"\n[Thông tin file] {file_path}")
            print(f"  - Số lượng bản ghi: {len(data)}")
            print(f"  - Các trường dữ liệu (keys): {list(sample.keys())}")
            
            # Gợi ý các trường máy sinh có sẵn (đối với DetectRL hoặc RAID)
            candidate_machine_fields = [k for k in sample.keys() if any(
                term in k for term in ['prompt', 'llm', 'machine', 'paraphrase', 'adversarial', 'gen']
            )]
            print(f"  - Các trường máy sinh tiềm năng: {candidate_machine_fields}")
    except Exception as e:
        print(f"[!] Lỗi đọc file: {e}")

def convert_file(
    input_file: str,
    output_name: Optional[str] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_pairs: Optional[int] = None,
    human_field: str = "human_text",
    machine_field: str = "direct_prompt",
    interleave: bool = True
) -> str:
    """
    Chuyển đổi một file JSON từ RAID hoặc DetectRL sang định dạng JSON Lines của PRDetect.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Không tìm thấy file: {input_file}")
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Xác định tên file đầu ra
    if not output_name:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        # Rút ngắn tiền tố nếu quá dài
        output_name = base_name.replace("_dataset", "").replace("_repetition_penalty", "")
        if machine_field != "direct_prompt":
            output_name += f"_{machine_field}"
            
    if not output_name.endswith(".json"):
        output_file = os.path.join(output_dir, f"{output_name}.json")
    else:
        output_file = os.path.join(output_dir, output_name)
        
    print(f"--> Đang đọc dữ liệu từ: {input_file} ...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if not isinstance(data, list):
        raise ValueError("Dữ liệu đầu vào phải là một JSON Array ([{...}, {...}])")
        
    if max_pairs is not None and max_pairs > 0:
        data = data[:max_pairs]
        
    human_count = 0
    machine_count = 0
    
    records_to_write = []
    
    for idx, item in enumerate(data):
        h_text = item.get(human_field, "")
        m_text = item.get(machine_field, "")
        
        # Nếu không có human_field trực tiếp, thử tìm 'abstract' hoặc các trường tương đương
        if not h_text and "abstract" in item:
            h_text = item.get("abstract", "")
            
        h_text = str(h_text).strip() if h_text else ""
        m_text = str(m_text).strip() if m_text else ""
        
        # Tạo bản ghi
        pair = []
        if h_text:
            pair.append({"text": h_text, "label": "human"})
            human_count += 1
            
        if m_text:
            pair.append({"text": m_text, "label": "machine"})
            machine_count += 1
            
        if interleave:
            # Xen kẽ: human rồi đến machine
            records_to_write.extend(pair)
        else:
            # Sẽ gom nhóm sau
            records_to_write.append(pair)
            
    # Ghi ra file theo định dạng JSON Lines
    with open(output_file, "w", encoding="utf-8") as out:
        if interleave:
            for rec in records_to_write:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            # Ghi toàn bộ human trước, sau đó toàn bộ machine
            humans = [p[0] for p in records_to_write if len(p) > 0 and p[0]["label"] == "human"]
            machines = [p[1] for p in records_to_write if len(p) > 1 and p[1]["label"] == "machine"]
            for h in humans:
                out.write(json.dumps(h, ensure_ascii=False) + "\n")
            for m in machines:
                out.write(json.dumps(m, ensure_ascii=False) + "\n")
                
    total_lines = human_count + machine_count
    print(f"[OK] Đã chuyển đổi thành công!")
    print(f"     - File đích: {output_file}")
    print(f"     - Mẫu Người viết (human): {human_count}")
    print(f"     - Mẫu Máy sinh ({machine_field}): {machine_count}")
    print(f"     - Tổng số dòng văn bản: {total_lines}")
    print(f"     - Để build graph: Thêm '{os.path.splitext(os.path.basename(output_file))[0]}' vào building_graph.py\n")
    return output_file

def main():
    parser = argparse.ArgumentParser(
        description="Công cụ chuyển đổi dữ liệu từ RAID / DetectRL sang định dạng chuẩn của PRDetect."
    )
    parser.add_argument(
        "--input", "-i", type=str,
        help="Đường dẫn đến file JSON của RAID hoặc DetectRL (VD: datasets/raid_dataset/xxx.json)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Tên file đầu ra (không cần đuôi .json, sẽ được lưu vào original_text/). Mặc định tự sinh theo tên file gốc."
    )
    parser.add_argument(
        "--output_dir", "-d", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Thư mục lưu file đầu ra. Mặc định là '{DEFAULT_OUTPUT_DIR}'."
    )
    parser.add_argument(
        "--max_samples", "-n", type=int, default=None,
        help="Số lượng cặp văn bản tối đa cần chuyển đổi (VD: -n 200 để lấy 200 cặp = 400 mẫu nhằm test nhanh)."
    )
    parser.add_argument(
        "--machine_field", "-m", type=str, default="direct_prompt",
        help="Trường chứa văn bản máy sinh. Mặc định: 'direct_prompt'. Các lựa chọn trong DetectRL: 'paraphrase_dipper_llm', 'adversarial_character_word_llm', 'paraphrase_polish_llm', v.v."
    )
    parser.add_argument(
        "--human_field", type=str, default="human_text",
        help="Trường chứa văn bản người viết. Mặc định: 'human_text'."
    )
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="Liệt kê danh sách các tập dữ liệu có sẵn trong thư mục datasets/."
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Kiểm tra cấu trúc và các trường (keys) của file đầu vào mà không chuyển đổi."
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_datasets()
        return
        
    if not args.input:
        parser.print_help()
        print("\n[MẸO] Chạy 'python convert_datasets.py --list' để xem danh sách các file có sẵn.")
        print("[VÍ DỤ] python convert_datasets.py -i datasets/raid_dataset/raid_test_dataset_llm_type_gpt2_decoding_type_greedy_repetition_penalty_no.json -n 100")
        return
        
    if args.inspect:
        inspect_dataset_fields(args.input)
        return
        
    convert_file(
        input_file=args.input,
        output_name=args.output,
        output_dir=args.output_dir,
        max_pairs=args.max_samples,
        human_field=args.human_field,
        machine_field=args.machine_field,
        interleave=True
    )

if __name__ == "__main__":
    main()
