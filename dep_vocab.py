"""
Quản lý từ điển quan hệ phụ thuộc cú pháp (Dependency Relation Vocabulary)
dành cho Mạng Tích chập Đồ thị Quan hệ (RGCN) trong PRDetect.
"""

import os
import json
from typing import Dict, List, Optional

# Danh sách chuẩn các quan hệ cú pháp phụ thuộc trong SpaCy (ClearNLP / Universal Dependencies)
CANONICAL_DEP_LABELS = [
    "<unk>",        # 0: Quan hệ không xác định / dự phòng
    "ROOT",         # 1: Gốc câu (root)
    "nsubj",        # 2: Chủ ngữ danh từ
    "nsubjpass",    # 3: Chủ ngữ bị động
    "dobj",         # 4: Tân ngữ trực tiếp
    "iobj",         # 5: Tân ngữ gián tiếp
    "pobj",         # 6: Tân ngữ của giới từ
    "prep",         # 7: Bổ ngữ giới từ
    "amod",         # 8: Bổ nghĩa tính từ
    "advmod",       # 9: Bổ nghĩa phó từ
    "compound",     # 10: Từ ghép
    "det",          # 11: Mạo từ / từ hạn định
    "punct",        # 12: Dấu câu
    "aux",          # 13: Trợ động từ
    "auxpass",      # 14: Trợ động từ bị động
    "cc",           # 15: Liên từ kết hợp
    "conj",         # 16: Thành phần liên hợp
    "ccomp",        # 17: Mệnh đề bổ ngữ
    "xcomp",        # 18: Mệnh đề bổ ngữ mở
    "relcl",        # 19: Mệnh đề quan hệ
    "advcl",        # 20: Mệnh đề trạng ngữ
    "acl",          # 21: Mệnh đề tính ngữ
    "poss",         # 22: Sở hữu
    "possessive",   # 23: Dấu sở hữu ('s)
    "case",         # 24: Dấu hiệu cách ngữ pháp
    "mark",         # 25: Từ nối mệnh đề phụ (marker)
    "appos",        # 26: Thành phần đồng vị
    "nummod",       # 27: Bổ nghĩa số từ
    "attr",         # 28: Thuộc tính
    "acomp",        # 29: Vị ngữ tính từ
    "cop",          # 30: Hệ từ (copula)
    "csubj",        # 31: Chủ ngữ mệnh đề
    "csubjpass",    # 32: Chủ ngữ mệnh đề bị động
    "agent",        # 33: Tác nhân trong câu bị động
    "neg",          # 34: Từ phủ định (not, n't)
    "prt",          # 35: Tiểu từ (phrasal verb particle)
    "expl",         # 36: Chủ ngữ giả (there, it)
    "parataxis",    # 37: Cú pháp song song
    "dep",          # 38: Quan hệ chưa phân loại
    "meta",         # 39: Yếu tố siêu ngôn ngữ
    "intj",         # 40: Thán từ
    "dative",       # 41: Bổ ngữ cách nhận (dative)
    "oprd",         # 42: Bổ ngữ tân ngữ
    "pcomp",        # 43: Bổ ngữ giới từ
    "quantmod",     # 44: Bổ nghĩa lượng từ
    "predet",       # 45: Từ hạn định trước
    "preconj",      # 46: Liên từ trước (either, both)
    "nn",           # 47: Danh từ bổ nghĩa (ClearNLP legacy)
    "nounmod",      # 48: Danh từ bổ ngữ
    "npadvmod",     # 49: Cụm danh từ làm trạng ngữ
    # Các nhãn Universal Dependencies mở rộng
    "nmod",         # 50: Bổ ngữ danh từ (UD)
    "obl",          # 51: Bổ ngữ xiên (oblique)
    "flat",         # 52: Tên riêng nhiều từ
    "fixed",        # 53: Cụm từ cố định
    "vocative",     # 54: Thành phần hô gọi
    "discourse",    # 55: Dấu hiệu đàm thoại
    "orphan",       # 56: Yếu tố tỉnh lược
    "goeswith",     # 57: Từ bị tách rời
    "list",         # 58: Danh sách liệt kê
    "dislocated"    # 59: Thành phần tách rời vị trí
]

DEFAULT_VOCAB_PATH = os.path.join(os.path.dirname(__file__), "dep_vocab.json")


class DependencyVocab:
    """
    Lớp quản lý bảng tra cứu quan hệ cú pháp phụ thuộc.
    Đảm bảo việc ánh xạ giữa nhãn string (ví dụ 'nsubj') và ID số nguyên (int)
    hoàn toàn nhất quán giữa quá trình trích xuất đồ thị, huấn luyện và suy luận.
    """
    def __init__(self, vocab_path: Optional[str] = None):
        self.vocab_path = vocab_path or DEFAULT_VOCAB_PATH
        self.dep2id: Dict[str, int] = {}
        self.id2dep: Dict[int, str] = {}
        self._init_vocab()

    def _init_vocab(self):
        # Nạp từ file nếu có sẵn, nếu không thì dùng CANONICAL_DEP_LABELS
        if os.path.exists(self.vocab_path):
            try:
                with open(self.vocab_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.dep2id = data.get("dep2id", {})
                self.id2dep = {int(k): v for k, v in data.get("id2dep", {}).items()}
                return
            except Exception as e:
                print(f"[Warning] Không thể nạp {self.vocab_path}: {e}. Khởi tạo từ danh mục mặc định.")

        for idx, label in enumerate(CANONICAL_DEP_LABELS):
            self.dep2id[label] = idx
            self.id2dep[idx] = label

    def encode(self, dep_label: str, allow_new: bool = False) -> int:
        """
        Chuyển nhãn quan hệ cú pháp sang ID số nguyên.
        Nếu không có và allow_new=False, trả về ID của '<unk>' (0).
        """
        dep_clean = dep_label.strip()
        if dep_clean in self.dep2id:
            return self.dep2id[dep_clean]
        
        # Thử với chữ thường
        dep_lower = dep_clean.lower()
        if dep_lower in self.dep2id:
            return self.dep2id[dep_lower]

        if allow_new:
            new_id = len(self.dep2id)
            self.dep2id[dep_clean] = new_id
            self.id2dep[new_id] = dep_clean
            self.save()
            return new_id

        return self.dep2id.get("<unk>", 0)

    def decode(self, dep_id: int) -> str:
        """Chuyển ID số nguyên sang nhãn quan hệ cú pháp"""
        return self.id2dep.get(dep_id, "<unk>")

    def __len__(self) -> int:
        return len(self.dep2id)

    @property
    def num_relations(self) -> int:
        """Tổng số quan hệ đã được đăng ký"""
        return len(self.dep2id)

    def save(self, target_path: Optional[str] = None):
        """Lưu bảng từ điển quan hệ ra file JSON"""
        path = target_path or self.vocab_path
        data = {
            "num_relations": len(self.dep2id),
            "dep2id": self.dep2id,
            "id2dep": self.id2dep
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# Khởi tạo singleton mặc định
_GLOBAL_VOCAB: Optional[DependencyVocab] = None


def get_default_vocab() -> DependencyVocab:
    global _GLOBAL_VOCAB
    if _GLOBAL_VOCAB is None:
        _GLOBAL_VOCAB = DependencyVocab()
        # Đảm bảo file dep_vocab.json tồn tại để tái sử dụng
        if not os.path.exists(_GLOBAL_VOCAB.vocab_path):
            _GLOBAL_VOCAB.save()
    return _GLOBAL_VOCAB
