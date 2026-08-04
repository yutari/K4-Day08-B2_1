# E-commerce Support RAG Chatbot

Tài liệu kỹ thuật của dự án chatbot hỏi đáp chính sách thương mại điện tử, xây dựng theo RAG (Retrieval-Augmented Generation). Tài liệu này mô tả đúng cách hệ thống đang vận hành, cách đọc điểm số trên UI và quy trình chạy/demo/evaluation.

## 1. Bài toán và mục tiêu

Người dùng cần tra cứu chính sách như thanh toán, trả hàng/hoàn tiền, giao hàng, quy định người bán và các hướng dẫn hỗ trợ. Các tài liệu này dài, nằm ở nhiều tệp và thường dùng ngôn ngữ khác với cách người dùng đặt câu hỏi.

Mục tiêu của hệ thống là:

- Tìm đúng đoạn tài liệu liên quan trước khi trả lời.
- Kết hợp tìm theo ý nghĩa và tìm theo từ khóa chính xác.
- Trả lời có citation theo tên tệp và mục tài liệu.
- Hiển thị nguồn web chính thức khi metadata có URL.
- Không bịa nguồn: nếu LLM không tạo được câu trả lời có citation hợp lệ thì chuyển sang bằng chứng trích xuất trực tiếp.
- Đánh giá định lượng Dense-only và Hybrid RAG bằng Golden Dataset.

Phạm vi kiến thức là corpus chính sách/hướng dẫn TMĐT trong repository. Đây không phải tư vấn pháp lý; nội dung trên website nguồn có thể thay đổi theo thời gian.

## 2. Kiến trúc tổng thể

~~~mermaid
flowchart LR
    A[Tài liệu PDF, JSON, Markdown] --> B[Chuẩn hóa Markdown]
    B --> C[Chunking + metadata]
    C --> D[ChromaDB + embedding]
    C --> E[BM25 index]

    U[Người dùng] --> M[Memory + Query Expansion]
    M --> F[Dense semantic search]
    M --> G[BM25 lexical search]
    F --> H[Weighted RRF]
    G --> H
    H --> I[Cross-Encoder reranking]
    I --> J{Dense confidence đủ ngưỡng?}
    J -- Có --> K[LLM grounded generation]
    J -- Không --> L[Structural PageIndex fallback]
    L --> K
    K --> V[Citation validator / extractive fallback]
    V --> W[Streamlit + source cards]
~~~

Luồng được tách thành ba nhóm:

| Nhóm | Thành phần | Vai trò |
|---|---|---|
| Ingestion | Task 1–4 | Thu thập, chuẩn hóa, chunk và index dữ liệu. |
| Retrieval | Task 5–9 | Dense search, BM25, fusion, rerank và fallback. |
| Answer & quality | Task 10, Streamlit, evaluation | Sinh câu trả lời, hậu kiểm citation, demo và A/B evaluation. |

## 3. Dữ liệu và chuẩn hóa

### 3.1. Nguồn dữ liệu

| Thư mục | Nội dung | Tình trạng corpus hiện tại |
|---|---|---|
| <code>data/landing/legal/</code> | Tài liệu PDF/DOCX chính sách gốc | 3 PDF nguồn |
| <code>data/landing/news/</code> | JSON/Markdown hướng dẫn và thông báo | 15 tệp dữ liệu |
| <code>data/standardized/</code> | Markdown đã chuẩn hóa cho retrieval | 18 Markdown, gồm 3 legal và 15 news |
| <code>chroma_db/</code> | ChromaDB persistent vector store | Sinh lại từ standardized corpus |

Task 1 chỉ xác thực sự tồn tại và kích thước hợp lệ của ít nhất 3 tài liệu PDF/DOCX thật; không tạo tài liệu chính sách giả. Task 2 dùng Crawl4AI khi có mạng, đồng thời có dữ liệu mẫu để pipeline vẫn có thể chạy trong môi trường demo offline.

### 3.2. Chuẩn hóa Markdown

Module <code>src/task3_convert_markdown.py</code> sử dụng MarkItDown để chuyển PDF/DOCX sang Markdown. JSON news được chuyển thành Markdown và có front matter để giữ provenance.

Metadata được bảo toàn hoặc suy ra gồm:

- <code>source</code>, <code>relative_path</code>, <code>title</code>, <code>type</code>
- <code>url</code>, <code>category</code>, <code>version</code>, <code>date</code>
- <code>customer_role</code>
- <code>chunk_index</code>, <code>section</code>

Task 4 cũng nhận diện metadata kiểu YAML cũ bị đặt sau tiêu đề Markdown. Khối này được tách khỏi nội dung chunk để các trường như <code>doc_id</code> hay <code>source_url</code> không trở thành bằng chứng trả lời.

## 4. Chunking, embedding và vector store

### 4.1. Chunking

Hệ thống dùng <code>RecursiveCharacterTextSplitter</code>.

| Tham số | Giá trị | Lý do |
|---|---:|---|
| Chunk size | 800 ký tự | Đủ chứa một quy định/điều kiện nhưng vẫn gọn để retrieval và LLM xử lý. |
| Chunk overlap | 100 ký tự | Giữ ngữ cảnh ở ranh giới giữa hai chunk. |
| Separator | Heading → paragraph → line → sentence → word | Ưu tiên cấu trúc Markdown và hạn chế cắt giữa ý. |

Mỗi chunk lưu heading gần nhất trong trường <code>section</code>. Đây là phần thứ hai của citation, ví dụ:

    [Nguồn: tra_hang_hoan_tien.md, Mục: Thời hạn gửi yêu cầu]

### 4.2. Embedding

| Thuộc tính | Cấu hình |
|---|---|
| Model | <code>sentence-transformers/all-MiniLM-L6-v2</code> |
| Số chiều | 384 |
| Chạy ở đâu | Local, không cần embedding API key |
| Normalization | L2-normalized |
| Vector store | ChromaDB persistent |
| Similarity | Cosine similarity |

Embedding giúp tìm các câu diễn đạt tương đương. Ví dụ, câu “đổi trả hàng trong bao lâu?” vẫn có thể khớp với đoạn “thời hạn gửi yêu cầu trả hàng/hoàn tiền” dù các từ không hoàn toàn giống nhau.

## 5. Retrieval pipeline

### 5.1. Dense semantic search

<code>src/task5_semantic_search.py</code> tạo embedding của query, truy vấn ChromaDB và trả về tối đa 20 chunk. Chroma dùng cosine distance nên hệ thống chuyển thành:

    dense_cosine_similarity = 1 - cosine_distance

Điểm dense được giới hạn trong khoảng 0–1 và sắp xếp giảm dần.

### 5.2. Lexical search với BM25

<code>src/task6_lexical_search.py</code> dùng BM25Okapi với tokenizer regex hỗ trợ Unicode/tiếng Việt. BM25 hữu ích cho:

- Số ngày, số tiền, điều kiện cụ thể.
- Tên sản phẩm, tên chính sách, mã hoặc thuật ngữ chính xác.
- Từ khóa như COD, ShopeePay, Sao Quả Tạ.

Khác biệt chính:

| Dense semantic search | BM25 lexical search |
|---|---|
| So khớp ý nghĩa bằng vector embedding | So khớp các token xuất hiện trong văn bản |
| Tốt khi người dùng diễn đạt khác tài liệu | Tốt khi cần đúng từ khóa, số liệu, tên riêng |
| Score là cosine similarity | Score là BM25 raw score, chỉ so sánh trong cùng một query |

### 5.3. Weighted RRF fusion

Dense và BM25 chạy song song bằng <code>ThreadPoolExecutor</code>, mỗi nhánh lấy tối đa 20 candidates. Kết quả được gộp bằng Weighted Reciprocal Rank Fusion:

    RRF(d) = Σ wᵢ / (k + rankᵢ(d)), với k = 60

Slider α trên sidebar là trọng số Dense:

    dense weight = α
    BM25 weight = 1 - α

RRF chỉ dùng thứ hạng nên không cộng trực tiếp cosine score với BM25 raw score, vốn thuộc hai thang đo khác nhau.

### 5.4. Cross-Encoder reranking và Cơ chế chấm điểm Reranking

Sau bước Weighted RRF Fusion, các ứng viên (candidates) tiếp tục được tái xếp hạng (rerank) bằng mô hình **Cross-Encoder**:

    cross-encoder/ms-marco-MiniLM-L-6-v2

#### Cơ chế hoạt động & So sánh với Bi-Encoder (Dense Search):
* **Bi-Encoder (ChromaDB Vector)**: Mã hóa Query và Chunk riêng biệt thành 2 vector độc lập, sau đó tính Cosine Similarity. Ưu điểm là rất nhanh (tra cứu trên HNSW index), nhưng bị mất đi sự tương tác chi tiết giữa các từ của Query và Context.
* **Cross-Encoder**: Đưa **đồng thời** cặp `(Query, Chunk)` qua các lớp Attention Layer của Transformer. Mô hình soi chiếu trực tiếp từng từ trong câu hỏi với từng từ trong văn bản, giúp đánh giá ngữ cảnh và mối quan hệ ngữ nghĩa tinh vi hơn nhiều.

#### Công thức chấm điểm & Chuẩn hóa:
1. **Raw CE Logit**: Cross-Encoder tính toán và trả về một điểm Logit không bị chặn (unbounded logit, ví dụ $+6.4811$ hoặc $-1.234$).
2. **Sigmoid Normalization**: Để hiển thị giao diện dưới dạng phần trăm (%), hệ thống áp dụng hàm Sigmoid:
   $$\text{CE relevance} = \sigma(\text{logit}) = \frac{1}{1 + e^{-\text{logit}}}$$
   *Ví dụ*: Logit $+6.4811 \xrightarrow{\text{Sigmoid}} 0.9985 \rightarrow 99.9\%$.
3. **Sắp xếp**: Danh sách kết quả cuối cùng được sắp xếp giảm dần theo điểm `normalized_score` (tương đương với thứ tự `cross_encoder_raw_score`). Nếu việc tải/chạy mô hình Cross-Encoder gặp lỗi, hệ thống sẽ tự động fallback về giữ nguyên thứ hạng `rrf_score`.

### 5.5. Confidence gate và PageIndex fallback

Confidence gate không dùng RRF hay Cross-Encoder. Nó là trung bình `dense_score` của candidates cuối, rồi so với `score_threshold` (mặc định 0.35).

- Confidence đủ ngưỡng: dùng hybrid candidates.
- Confidence thấp hoặc không có candidate: dùng `src/task8_pageindex_vectorless.py`.

“PageIndex” trong dự án này là structural fallback local, không phải PageIndex cloud SDK. Nó duyệt Markdown theo heading/section, đo keyword coverage của query và trả về section phù hợp. Vì chạy local nên không gửi corpus ra dịch vụ bên ngoài.

## 6. Cách đọc score trên giao diện

Các score không cùng ý nghĩa và không nên so sánh trực tiếp.

| Nhãn UI | Nguồn | Thang đo | Cách diễn giải đúng |
|---|---|---|---|
| Dense cosine | ChromaDB | 0–100% | Độ tương đồng embedding. Dùng cho confidence gate. Không phải xác suất đúng tuyệt đối. |
| BM25 raw | BM25Okapi | Số dương không bị chặn | Chỉ so sánh các kết quả của cùng một query. |
| RRF rank | Weighted RRF | Số nhỏ, thường khoảng 0.01 | Điểm fusion theo vị trí xếp hạng; không phải %. |
| CE relevance | Cross-Encoder | 0–100% sau sigmoid | Tín hiệu reranking để đọc dễ hơn, không phải xác suất/độ chính xác đã calibration. |
| Keyword coverage | Structural fallback | 0–100% | Tỷ lệ token query xuất hiện trong section. |

Cross-Encoder gốc trả về logit không bị chặn. UI lưu logit gốc và hiển thị thêm bản chuẩn hóa:

    CE relevance = sigmoid(raw CE logit)

Ví dụ logit 6.4811 sẽ thành 99.85%, hiển thị 99.9%. Điều đó **không có nghĩa** tài liệu hoặc câu trả lời đúng 99.9%; sigmoid bão hòa nhanh với logit lớn. Khi demo nên nói:

> CE relevance là điểm xếp hạng được chuẩn hóa để quan sát; không phải confidence đã được hiệu chuẩn. Hãy xem Dense cosine, RRF/BM25 và citation cùng nhau.

Muốn có “confidence %” thực sự cần calibration trên tập gán nhãn độc lập, ví dụ Platt scaling/isotonic regression sau khi thu thập relevance labels.

## 7. Query expansion, memory và xử lý chào hỏi / tâm sự (Small Talk)

Streamlit lưu tối đa 20 messages (10 lượt hội thoại gần nhất). Với câu hỏi ngắn/phụ thuộc ngữ cảnh, `src/task09_query_expansion.py` ghép câu hỏi trước đó vào query retrieval. Khi generation, chỉ 6 messages gần nhất được gửi vào prompt để giới hạn context.

Ví dụ:

    Câu trước: Thời hạn trả hàng là bao lâu?
    Câu sau: Có mất phí không?
    Query retrieval: Thời hạn trả hàng là bao lâu? — Câu hỏi tiếp theo: Có mất phí không?

Các câu chào hỏi, cảm ơn, hỏi tên bot hoặc tâm sự cảm xúc tự do (như `"xin chào"`, `"hello"`, `"cảm ơn"`, `"chán quá"`, `"mệt quá"`, `"bạn là ai"`...) được nhận diện trước qua bộ phân loại Intent / Small Talk:
- **Ngắt ghép ngữ cảnh**: Không bị Query Expansion tự động ghép với câu hỏi chính sách trước đó.
- **Bỏ qua Retrieval**: Tắt hoàn toàn luồng tìm kiếm tài liệu Shopee/TMĐT để không gửi trích dẫn tài liệu ngẫu nhiên/không liên quan.
- **Phản hồi thân thiện**: Chatbot đáp lời xã giao/đồng cảm và nhắc lại đúng phạm vi hỗ trợ của hệ thống.

## 8. Generation, citation và an toàn

### 8.1. Grounded generation

<code>src/task10_generation.py</code> yêu cầu LLM:

1. Chỉ dùng context đã truy xuất.
2. Không có bằng chứng thì trả lời không thể xác minh.
3. Gắn citation ngay sau mệnh đề có thông tin.
4. Trả lời bằng tiếng Việt.

Thứ tự failover provider là Gemini → OpenRouter → OpenAI. Không có API key vẫn chạy được ở chế độ extractive fallback.

### 8.2. Hậu kiểm citation

Citation hợp lệ phải có đúng cả source và section của một retrieved chunk:

    [Nguồn: ten_file.md, Mục: ten_heading]

Sau khi LLM trả lời, validator:

- Chuẩn hóa biến thể Source/Section về đúng nhãn tiếng Việt.
- Từ chối tên tệp hoặc mục không có trong context.
- Từ chối đoạn nội dung dài không kèm citation.
- Từ chối output có front matter/metadata tài liệu.

| Trạng thái | Ý nghĩa |
|---|---|
| <code>validated</code> | LLM trả lời và citation đã được hậu kiểm. |
| <code>extractive_fallback</code> | LLM thiếu/sai citation; hệ thống hiển thị bằng chứng đã làm sạch, có citation chuẩn. |
| <code>no_evidence</code> | Không có context phù hợp để trả lời. |
| <code>not_required</code> | Chào hỏi/cảm ơn, không phải câu hỏi chính sách. |

Extractive fallback loại heading Markdown, YAML/front matter, các trường như <code>doc_id</code>, <code>source_url</code> và <code>title</code>. Vì vậy metadata không bị render thành tiêu đề lớn trong chat.

Khi fallback còn evidence, hệ thống trả tối đa 3 chunks và cắt mỗi đoạn ở 420 ký tự, sau đó gắn citation chuẩn. Nếu không có evidence nào, hệ thống trả câu không thể xác minh thay vì tự suy đoán.

### 8.3. Source cards

Mỗi source card hiển thị:

- Tên tài liệu, loại tài liệu và section.
- URL nguồn chính thức nếu metadata có URL.
- Score đúng loại và các score liên quan.
- Nội dung chunk.

URL được ưu tiên theo metadata trong index; nếu index cũ thiếu URL, UI thử đọc Markdown gốc và cuối cùng dùng map URL cho các tệp legacy.

## 9. Điều khiển trên Streamlit

| Điều khiển | Tác động thật |
|---|---|
| Số lượng chunks (top_k) | Số chunk giữ lại sau reranking/fallback để tạo context. |
| Ngưỡng Score Threshold | Ngưỡng trung bình Dense cosine để quyết định dùng hybrid hay structural fallback. |
| Ưu tiên Semantic Search (α) | Trọng số Dense trong Weighted RRF; phần còn lại dành cho BM25. |
| Sử dụng Cross-Encoder Reranking | Bật/tắt bước Cross-Encoder sau fusion. |
| Xóa lịch sử hội thoại | Xóa memory của phiên Streamlit hiện tại. |

Sau khi thay đổi tài liệu, logic chunking hoặc metadata, cần re-index trước khi kiểm tra UI để ChromaDB không còn dữ liệu cũ.

## 10. Evaluation A/B

### 10.1. Golden Dataset và cấu hình

File <code>group_project/evaluation/golden_dataset.json</code> có 16 câu hỏi với:

- <code>question</code>
- <code>expected_answer</code>
- <code>expected_context</code>

Hai nhánh được so sánh:

| Config | Retrieval | Generation policy |
|---|---|---|
| A | Dense-only ChromaDB | Cùng policy generation/citation với Config B |
| B | Dense + BM25 + Weighted RRF + Cross-Encoder + fallback | Cùng policy generation/citation với Config A |

### 10.2. Bốn metric RAGAS

Khi có evaluator API, script dùng RAGAS 0.1-compatible với:

| Metric | Đo lường |
|---|---|
| Faithfulness | Câu trả lời có bám bằng chứng context không. |
| Answer Relevancy | Câu trả lời có giải quyết đúng câu hỏi không. |
| Context Recall | Context truy xuất có bao phủ evidence/ground truth không. |
| Context Precision | Các context được ưu tiên có thực sự hữu ích không. |

Report xuất A/B metrics, chênh lệch từng metric và 5 Worst Performers của Config B cùng nguyên nhân/khuyến nghị.

### 10.3. RAGAS thật và offline proxy

Hai loại kết quả này phải được phân biệt:

| Chế độ | Lệnh | Ý nghĩa |
|---|---|---|
| RAGAS thật | <code>python group_project/evaluation/eval_pipeline.py --require-ragas</code> | Chỉ thành công khi evaluator/key/model sẵn sàng; phù hợp để nộp/chấm. |
| Offline proxy | <code>python group_project/evaluation/eval_pipeline.py --offline</code> | Cosine proxy local để demo; **không phải** điểm RAGAS. |

Không truyền flag sẽ thử RAGAS trước và fallback sang offline proxy nếu không khả dụng; report sẽ gắn nhãn backend thực tế. File <code>results.md</code> hiện phải được đọc theo nhãn backend của lần chạy, không được gọi mọi con số trong đó là RAGAS nếu nó ghi “Offline cosine proxy”.

## 11. Cài đặt và chạy

README khuyến nghị Python 3.10 hoặc 3.11. Tại thư mục dự án:

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
~~~

Cấu hình ít nhất một provider để có generation tự nhiên; nếu không chatbot vẫn dùng extractive fallback:

~~~text
GEMINI_API_KEY=...
# hoặc OPENROUTER_API_KEY=...
# hoặc OPENAI_API_KEY=...
~~~

Để chạy RAGAS thật, đặt thêm:

~~~text
RAGAS_EVALUATOR_API_KEY=...
RAGAS_EVALUATOR_MODEL=gpt-4o-mini
RAGAS_EVALUATOR_EMBEDDING_MODEL=text-embedding-3-small
# RAGAS_EVALUATOR_BASE_URL=...  # chỉ khi dùng endpoint tương thích OpenAI
~~~

Thứ tự chạy đầy đủ:

~~~powershell
# Kiểm tra / tạo dữ liệu
python -m src.task1_collect_legal_docs
python -m src.task2_crawl_news
python -m src.task3_convert_markdown

# Rebuild index sau khi standardized corpus hoặc metadata thay đổi
python -m src.task4_chunking_indexing

# Kiểm thử pipeline
pytest tests/ -v

# Chạy chatbot
streamlit run app.py

# Chạy benchmark thật
python group_project/evaluation/eval_pipeline.py --require-ragas
~~~

## 12. Kịch bản demo đề xuất

1. Mở Streamlit, cho thấy sidebar: top_k, threshold, α và reranking.
2. Hỏi một câu semantic, ví dụ “Thời hạn yêu cầu trả hàng/hoàn tiền là bao lâu?”.
3. Mở source cards, kiểm tra citation, section và link nguồn.
4. Hỏi câu follow-up “Có mất phí không?” để minh họa memory/query expansion.
5. Chỉnh α hoặc tắt reranking và quan sát thay đổi thứ hạng source.
6. Mở tab Evaluation để trình bày A/B, backend metric và Worst Performers.
7. Giải thích score: CE relevance dùng cho xếp hạng, không phải xác suất đúng tuyệt đối.

## 13. Kiểm thử và xử lý sự cố

| Hiện tượng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| Không có kết quả Dense/BM25 | Chưa build index hoặc corpus trống | Chạy lại Task 3 rồi Task 4. |
| Source card không có link | Index cũ thiếu metadata URL | Re-index; UI vẫn có fallback từ Markdown/map URL. |
| Chat hiển thị metadata hoặc heading lớn | Chunk lấy từ index cũ hoặc LLM output không hợp lệ | Chạy lại Task 4; citation validator và extractive cleaner đã chặn output mới. |
| CE relevance gần 100% | Sigmoid của Cross-Encoder logit bão hòa | Đọc raw CE logit, Dense cosine và citation; không coi đó là probability. |
| Evaluation báo Offline cosine proxy | Thiếu evaluator key/RAGAS không khả dụng | Cấu hình evaluator rồi chạy với <code>--require-ragas</code>. |
| Virtualenv báo không tìm thấy Python | Môi trường được tạo từ Python đã bị gỡ | Tạo môi trường mới bằng <code>py -3.11 -m venv .venv311</code>, kích hoạt nó và cài lại requirements. |

Các regression test bổ sung:

- <code>tests/test_citation_validation.py</code>: citation bịa/sai/mất citation phải bị từ chối.
- <code>tests/test_generation_fallback.py</code>: chào hỏi không vào RAG; fallback không lộ front matter.
- <code>tests/test_score_semantics.py</code>: tách Cross-Encoder raw logit, normalized display và RRF score.

## 14. Quyền riêng tư, giới hạn và hướng nâng cấp

### 14.1. Quyền riêng tư

Embedding, BM25, ChromaDB và structural fallback chạy local. Tuy nhiên, khi cấu hình Gemini, OpenRouter hoặc OpenAI, context được truy xuất sẽ được gửi tới provider tương ứng để tạo câu trả lời. Khi chạy RAGAS thật, question, answer, context và ground truth cũng được gửi tới evaluator đã cấu hình.

Không đưa API key vào source code, ảnh chụp màn hình hoặc tài liệu nộp bài. Chỉ đặt chúng trong <code>.env</code>, và giữ <code>.env</code> ngoài version control.

### 14.2. Giới hạn và roadmap

| Hiện tại | Hướng nâng cấp |
|---|---|
| all-MiniLM-L6-v2 nhẹ nhưng không chuyên biệt tiếng Việt | Thử embedding multilingual như BGE-M3 và đánh giá lại trên Golden Dataset. |
| BM25 dùng tokenizer regex đơn giản | Thêm synonym dictionary, word segmentation hoặc query translation. |
| CE relevance chưa được calibration | Thu thập relevance labels và calibration riêng. |
| Structural PageIndex là local fallback | Tích hợp PageIndex SDK thật nếu dự án cần cloud index. |
| Corpus có thể cũ so với website nguồn | Đặt lịch crawl/re-index và lưu version/ngày thu thập. |
| Evaluation cần evaluator API để có RAGAS thật | Giữ strict mode trong CI trước khi xuất report nộp bài. |

## 15. Mapping Task 1–10 và bản đồ mã nguồn

| Task | Deliverable chính | Module |
|---:|---|---|
| 1 | Kiểm tra tối thiểu 3 tài liệu chính sách nguồn | <code>task1_collect_legal_docs.py</code> |
| 2 | Crawl/lưu bài hướng dẫn, có offline sample fallback | <code>task2_crawl_news.py</code> |
| 3 | Chuẩn hóa dữ liệu sang Markdown | <code>task3_convert_markdown.py</code> |
| 4 | Chunking, embedding và ChromaDB indexing | <code>task4_chunking_indexing.py</code> |
| 5 | Dense semantic search | <code>task5_semantic_search.py</code> |
| 6 | BM25 lexical search | <code>task6_lexical_search.py</code> |
| 7 | Weighted RRF và Cross-Encoder reranking | <code>task7_reranking.py</code> |
| 8 | Structural/PageIndex local fallback | <code>task8_pageindex_vectorless.py</code> |
| 9 | Hybrid retrieval, confidence gate và fallback logic | <code>task9_retrieval_pipeline.py</code> |
| 10 | Generation, citation validation và extractive fallback | <code>task10_generation.py</code> |

| File | Vai trò |
|---|---|
| <code>app.py</code> | Streamlit UI, controls, source cards và hiển thị score. |
| <code>src/task1_collect_legal_docs.py</code> | Kiểm tra tài liệu pháp lý nguồn. |
| <code>src/task2_crawl_news.py</code> | Crawl/tạo dữ liệu hỗ trợ. |
| <code>src/task3_convert_markdown.py</code> | Chuẩn hóa landing data sang Markdown. |
| <code>src/task4_chunking_indexing.py</code> | Chunking, embedding, metadata và ChromaDB indexing. |
| <code>src/task5_semantic_search.py</code> | Dense cosine search. |
| <code>src/task6_lexical_search.py</code> | BM25 lexical search. |
| <code>src/task7_reranking.py</code> | Weighted RRF và Cross-Encoder reranking. |
| <code>src/task8_pageindex_vectorless.py</code> | Structural local fallback. |
| <code>src/task9_retrieval_pipeline.py</code> | Orchestration retrieval và confidence gate. |
| <code>src/task10_generation.py</code> | Grounded generation, citation validation và extractive fallback. |
| <code>group_project/evaluation/eval_pipeline.py</code> | A/B evaluation, RAGAS và report. |

## 16. Kết luận

Đây là RAG chatbot hybrid: embedding/ChromaDB giải quyết tương đồng ngữ nghĩa, BM25 bảo vệ các từ khóa chính xác, Weighted RRF kết hợp hai tín hiệu, Cross-Encoder tinh chỉnh thứ hạng, còn structural fallback và citation validator bảo vệ hệ thống khi retrieval/generation không chắc chắn. Mọi câu trả lời chính sách cần được đọc cùng citation và link nguồn, thay vì chỉ dựa vào một score hiển thị trên giao diện.
