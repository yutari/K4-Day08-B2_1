"""
Task 1 — Thu thập văn bản chính sách thương mại điện tử / hỗ trợ khách hàng.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản chính sách (PDF/DOCX) từ trang chính thức của một sàn TMĐT.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, mô tả đúng nội dung.

Gợi ý nguồn (ví dụ trang công khai Shopee Vietnam — help.shopee.vn):
    - https://help.shopee.vn/portal/4/article/77251 (Chính sách trả hàng và hoàn tiền)
    - https://help.shopee.vn/portal/4/article/79198 (Phương thức thanh toán)
    - https://help.shopee.vn/portal/4/article/77244 (Chính sách bảo mật)

Gợi ý văn bản (chủ đề chính sách thương mại điện tử):
    - Chính sách đổi trả/hoàn tiền (Returns/Refund Policy)
    - Phương thức thanh toán (Payment Methods)
    - Chính sách bảo mật (Privacy Policy)
    - Quy định đăng bán sản phẩm cho người bán (Seller Listing Regulations)

Nhớ gắn metadata `customer_role` (`buyer`/`seller`/`both`) cho từng tài liệu — yêu cầu riêng
của K4 Variant (kế thừa từ Lab 07), cần thiết để viết benchmark query dùng metadata_filter.

Lưu ý: một số trang help center dùng JavaScript render nội dung (SPA) — crawl về chỉ thấy
tiêu đề mà không có nội dung thật. Đổi sang bài viết khác cùng domain thay vì cố xử lý,
và chỉ dùng nguồn công khai/được phép chia sẻ.
"""

from pathlib import Path
from fpdf import FPDF

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def generate_sample_pdfs():
    """Tạo 3 file PDF mẫu bằng fpdf2."""
    policies = [
        {
            "filename": "chinh-sach-tra-hang.pdf",
            "title": "Chinh sach tra hang va hoan tien",
            "content": "1. Dieu kien tra hang: Khach hang co the tra hang trong vong 7 ngay.\n2. Quy trinh hoan tien: Tien se duoc hoan vao vi trong 24h.",
            "role": "both"
        },
        {
            "filename": "phuong-thuc-thanh-toan.pdf",
            "title": "Phuong thuc thanh toan",
            "content": "1. Thanh toan khi nhan hang (COD).\n2. Thanh toan qua the tin dung/ghi no.\n3. Thanh toan qua vi dien tu.",
            "role": "buyer"
        },
        {
            "filename": "quy-dinh-nguoi-ban.pdf",
            "title": "Quy dinh danh cho nguoi ban",
            "content": "1. Khong dang ban hang gia, hang nhai.\n2. Thoi gian chuan bi hang toi da 2 ngay.\n3. Phi san giao dich la 3%.",
            "role": "seller"
        }
    ]

    for policy in policies:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, policy["title"], ln=True, align="C")
        pdf.ln(10)
        pdf.set_font("helvetica", "", 12)
        pdf.multi_cell(0, 10, policy["content"])
        
        filepath = DATA_DIR / policy["filename"]
        pdf.output(str(filepath))
        print(f"✓ Đã tạo file: {filepath} (Metadata role: {policy['role']})")


if __name__ == "__main__":
    setup_directory()
    generate_sample_pdfs()
