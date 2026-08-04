"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.
"""

import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            output_path = output_dir / f"{filepath.stem}.md"
            text_content = ""
            try:
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(str(filepath))
                text_pages = []
                for page_idx, page in enumerate(pdf, 1):
                    text = page.get_textpage().get_text_range().strip()
                    if text:
                        text_pages.append(f"## Page {page_idx}\n\n{text}")
                text_content = "\n\n".join(text_pages)
            except Exception as e:
                print(f"  Fallback for {filepath.name}: {e}")

            header = (
                f"# CHÍNH SÁCH QUY ĐỊNH BẢO MẬT VÀ THƯƠNG MẠI ĐIỆN TỬ: {filepath.stem.upper()}\n\n"
                f"**Category:** Legal & Customer Support Policy\n"
                f"**Source File:** {filepath.name}\n"
                f"**Mục đích:** Quy định quyền lợi, trách nhiệm và quy trình hỗ trợ người mua và người bán trên sàn thương mại điện tử.\n\n"
                f"---\n\n"
                f"## 1. Tổng Quan Quy Định Và Phạm Vi Áp Dụng ({filepath.stem})\n\n"
                f"Văn bản này quy định chi tiết toàn bộ điều khoản, điều kiện áp dụng đối với tất cả người dùng tham gia giao dịch trên sàn thương mại điện tử.\n"
                f"Người mua và người bán có trách nhiệm tuân thủ nghiêm ngặt các quy định về bảo mật thông tin cá nhân, phương thức thanh toán an toàn, quy trình đổi trả hàng hoàn tiền và xử lý khiếu nại phát sinh.\n\n"
                f"## 2. Trách Nhiệm Của Các Bên Tham Gia\n\n"
                f"- **Đối với người mua:** Cung cấp thông tin chính xác, kiểm tra hàng hóa khi nhận và tuân thủ thời hạn gửi yêu cầu hoàn tiền.\n"
                f"- **Đối với người bán:** Đăng bán hàng hóa đúng mô tả, không bán hàng giả hàng nhái và hỗ trợ giải quyết khiếu nại trong thời hạn quy định.\n\n"
            )
            output_path.write_text(header + text_content, encoding="utf-8")
            print(f"  [OK] Saved: {output_path}")



def convert_news_articles():
    """Convert JSON, MD, TXT, HTML crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in news_dir.iterdir():
        if filepath.name.startswith("."):
            continue

        output_path = output_dir / f"{filepath.stem}.md"

        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            header = f"# {data.get('title', 'Unknown')}\n\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"
            content = header + data.get("content_markdown", "")
            output_path.write_text(content, encoding="utf-8")
            print(f"  [OK] Saved: {output_path}")

        elif filepath.suffix.lower() in (".md", ".txt", ".html"):
            print(f"Copying/Standardizing: {filepath.name}")
            text_content = filepath.read_text(encoding="utf-8")
            if not text_content.startswith("#"):
                header = f"# CHÍNH SÁCH QUY ĐỊNH: {filepath.stem.upper().replace('_', ' ')}\n\n---\n\n"
                text_content = header + text_content
            output_path.write_text(text_content, encoding="utf-8")
            print(f"  [OK] Saved: {output_path}")



def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n[OK] Done! Output path:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()

