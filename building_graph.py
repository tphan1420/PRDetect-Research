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

def build_graph(json_texts):
    init_models()
    import torch
    from tqdm import tqdm
    start_time = time.time()
    texts = list()
    y = list()
    for json_text in json_texts:
        texts.append(json.loads(json_text)['text'])
        label = 1 if "human" in json.loads(json_text)['label'] else 0
        y.append(label)
    y = torch.tensor(y, dtype=torch.float32)
    tokenized_sentences = list()
    all_token_embeddings = list()
    all_edge_index = list()
    all_sparse_adj_matrix = list()
    for text in tqdm(texts):
        try:
            doc = nlp(text)
            tokenized_sentence = [token.text for token in doc]
            tokenized_sentences.append(tokenized_sentence)
            # print(tokenized_sentence)
            
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
            token_embeddings = torch.cat(chunk_outputs, dim=0)
            all_token_embeddings.append(token_embeddings)
            # print(len(tokenized_sentence))
            # print(token_embeddings.shape)
            node_relations = list()
            for word in doc:        
                node_relations.append([word.i,word.head.i])
                # Add self-loops
                # if word.i != word.head.i:
                #     node_relations.append([word.i,word.i])
            edge0 = list()
            edge1 = list()
            for edge in node_relations:
                edge0.append(edge[0])
                edge1.append(edge[1])
            edge_index = torch.tensor([edge0, edge1], dtype=torch.long)
            all_edge_index.append(edge_index)
            # sparse_adj_matrix = csr_matrix((np.ones(len(edge0)),(np.array(edge0), np.array(edge1))),shape=(len(tokenized_sentence),len(tokenized_sentence)))
            # dependency_matrix = sparse_adj_matrix
            # print(sparse_adj_matrix)
            # all_sparse_adj_matrix.append(sparse_adj_matrix)
        except Exception as e:
            print(text)
            print(e)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    return all_token_embeddings, all_edge_index, y

def read_json(file_name):
    # Loại bỏ đuôi .json nếu người dùng truyền vào
    if file_name.endswith(".json"):
        file_name = file_name[:-5]
    if os.path.exists(f"original_text/{file_name}.json"):
        target_path = f"original_text/{file_name}.json"
    elif os.path.exists(f"{file_name}.json"):
        target_path = f"{file_name}.json"
    else:
        target_path = f"original_text/{file_name}.json"

    texts = list()
    with open(target_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if line.strip():
                texts.append(line)
    return texts

def save_pkl(file_name, all_token_embeddings, all_edge_index, y):
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    os.makedirs("./graph_data", exist_ok=True)
    out_path = f"./graph_data/{base_name}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"all_token_embeddings": all_token_embeddings,
                     "all_edge_index": all_edge_index,
                     "y": y}, f)
    print(f"[OK] Đã lưu dữ liệu đồ thị thành công tại: {out_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Xây dựng Cây cú pháp và Đồ thị đặc trưng (Graph Data) cho PRDetect")
    parser.add_argument(
        "--file", "-f", nargs="+", default=None,
        help="Tên file hoặc danh sách các file trong original_text/ cần build graph (VD: --file raid_gpt2_test hoặc -f f1 f2)"
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
        base_name = os.path.splitext(os.path.basename(file))[0]
        print(f"\n[Processing] Đang xử lý xây dựng đồ thị cho: {base_name} ...")
        json_data = read_json(file)
        print(f"-> Đã đọc {len(json_data)} mẫu văn bản từ original_text/{base_name}.json")
        all_token_embeddings, all_edge_index, y = build_graph(json_data)
        save_pkl(base_name, all_token_embeddings, all_edge_index, y)