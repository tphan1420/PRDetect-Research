import sys
import os
import time
import json
import pickle

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

nlp = None
model = None
tokenizer = None
device = None

def init_models():
    global nlp, model, tokenizer, device
    if nlp is not None and model is not None:
        return
    import spacy
    import torch
    from transformers import RobertaTokenizer, RobertaModel
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Đang nạp mô hình SpaCy (en_core_web_sm) và RoBERTa trên thiết bị: {device} ...")
    nlp = spacy.load("en_core_web_sm")
    
    if os.path.exists("./roberta-base/vocab.json"):
        model = RobertaModel.from_pretrained("./roberta-base/").to(device)
        tokenizer = RobertaTokenizer("./roberta-base/vocab.json", "./roberta-base/merges.txt", use_fast=False)
    else:
        model = RobertaModel.from_pretrained("roberta-base").to(device)
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=False)

def build_graph(json_texts, return_edge_type=True, bidirectional=False):
    """
    Xây dựng cây cú pháp phụ thuộc và biểu diễn nhúng node cho tập văn bản.
    
    Tham số:
        json_texts: Danh sách chuỗi JSON {"text": ..., "label": ...}
        return_edge_type: Nếu True, trả về thêm all_edge_type cho RGCN.
        bidirectional: Nếu True, thêm cạnh đảo chiều và nhãn quan hệ đảo tương ứng.
    
    Trả về:
        all_token_embeddings, all_edge_index, [all_edge_type], y
    """
    init_models()
    import torch
    from tqdm import tqdm
    from dep_vocab import get_default_vocab
    
    vocab = get_default_vocab()
    start_time = time.time()
    texts = list()
    y = list()
    for json_text in json_texts:
        data = json.loads(json_text)
        texts.append(data['text'])
        lbl = data.get('label', '')
        if isinstance(lbl, (int, float)):
            label = int(lbl)
        else:
            label = 1 if "human" in str(lbl).lower() else 0
        y.append(label)
    y = torch.tensor(y, dtype=torch.float32)

    tokenized_sentences = list()
    all_token_embeddings = list()
    all_edge_index = list()
    all_edge_type = list()
    valid_indices = list()

    for idx, text in enumerate(tqdm(texts, desc="Building Graph")):
        try:
            doc = nlp(text)
            tokenized_sentence = [token.text for token in doc]
            if len(tokenized_sentence) == 0:
                continue
            tokenized_sentences.append(tokenized_sentence)
            
            max_length = 512
            chunks = [tokenized_sentence[i:i+max_length] for i in range(0, len(tokenized_sentence), max_length)]
            chunk_outputs = []
            for chunk in chunks:
                token_ids = tokenizer.convert_tokens_to_ids(chunk)
                input_ids = torch.tensor(token_ids).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(input_ids)

                last_hidden_states = output.last_hidden_state
                token_embeddings = last_hidden_states[0]
                chunk_outputs.append(token_embeddings)
            token_embeddings = torch.cat(chunk_outputs, dim=0).cpu()

            edge0 = list()
            edge1 = list()
            edge_types = list()
            for word in doc:
                # Cạnh phụ thuộc: dependent -> head
                edge0.append(word.i)
                edge1.append(word.head.i)
                rel_id = vocab.encode(word.dep_)
                edge_types.append(rel_id)

                if bidirectional and word.i != word.head.i:
                    # Cạnh ngược: head -> dependent với quan hệ đảo
                    edge0.append(word.head.i)
                    edge1.append(word.i)
                    rev_rel_id = vocab.encode(f"{word.dep_}_rev", allow_new=True)
                    edge_types.append(rev_rel_id)

            edge_index = torch.tensor([edge0, edge1], dtype=torch.long)
            edge_type = torch.tensor(edge_types, dtype=torch.long)

            all_token_embeddings.append(token_embeddings)
            all_edge_index.append(edge_index)
            all_edge_type.append(edge_type)
            valid_indices.append(idx)
        except Exception as e:
            print(f"[!] Lỗi tại mẫu #{idx}: {e}")

    if len(valid_indices) < len(y):
        y = y[valid_indices]

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds | Tổng mẫu hợp lệ: {len(all_edge_index)}")

    if return_edge_type:
        return all_token_embeddings, all_edge_index, all_edge_type, y
    return all_token_embeddings, all_edge_index, y

def read_json(file_name):
    # Loại bỏ đuôi .json nếu người dùng truyền vào
    clean_name = file_name[:-5] if file_name.endswith(".json") else file_name

    candidates = [
        os.path.join("original_text", f"{clean_name}.json"),
        os.path.join("perturbed_text", f"{clean_name}.json"),
        f"{clean_name}.json",
        file_name
    ]

    target_path = None
    for cand in candidates:
        if os.path.exists(cand):
            target_path = cand
            break

    if target_path is None:
        raise FileNotFoundError(
            f"Không tìm thấy file '{file_name}' trong original_text/ hoặc perturbed_text/!"
        )

    print(f"-> Nạp văn bản từ: {target_path}")
    texts = list()
    with open(target_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if line.strip():
                texts.append(line)
    return texts

def save_pkl(file_name, all_token_embeddings, all_edge_index, y, all_edge_type=None):
    clean_name = file_name[:-5] if file_name.endswith(".json") else file_name
    base_name = os.path.basename(clean_name)
    os.makedirs("./graph_data", exist_ok=True)
    out_path = f"./graph_data/{base_name}.pkl"
    save_dict = {
        "all_token_embeddings": all_token_embeddings,
        "all_edge_index": all_edge_index,
        "y": y
    }
    if all_edge_type is not None:
        save_dict["all_edge_type"] = all_edge_type
    with open(out_path, "wb") as f:
        pickle.dump(save_dict, f)
    has_rel = "CÓ quan hệ đa loại (RGCN)" if all_edge_type is not None else "không có quan hệ (GCN thuần)"
    print(f"[OK] Đã lưu dữ liệu đồ thị thành công tại: {out_path} ({has_rel})")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Xây dựng Cây cú pháp Đa quan hệ (RGCN) và Đồ thị đặc trưng cho PRDetect")
    parser.add_argument(
        "--file", "-f", nargs="+", default=None,
        help="Tên file hoặc danh sách các file trong original_text/ cần build graph (VD: --file raid_gpt2_test hoặc -f f1 f2)"
    )
    parser.add_argument(
        "--bidirectional", "-b", action="store_true",
        help="Thêm cạnh hai chiều và quan hệ đảo (inverse relations) cho cây cú pháp."
    )
    args = parser.parse_args()

    if args.file:
        files = args.file
    else:
        print("Không có tham số --file, sử dụng danh sách mặc định: hc3_train, hc3_val, hc3_test")
        files = [
            "hc3_train",
            "hc3_val",
            "hc3_test"
        ]

    for file in files:
        clean_name = file[:-5] if file.endswith(".json") else file
        base_name = os.path.basename(clean_name)
        print(f"\n[Processing] Đang xử lý xây dựng đồ thị cho: {base_name} ...")
        json_data = read_json(file)
        print(f"-> Đã đọc {len(json_data)} mẫu văn bản từ file {base_name}")
        all_token_embeddings, all_edge_index, all_edge_type, y = build_graph(
            json_data, return_edge_type=True, bidirectional=args.bidirectional
        )
        save_pkl(base_name, all_token_embeddings, all_edge_index, y, all_edge_type=all_edge_type)