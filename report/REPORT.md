# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** T4 GPU

**Python / PyTorch / Ultralytics:** Python 3.13.16 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`= 0.510915 ; `taxonomy_name`= ImageNet-1K
2. Record này nói về **toàn bộ bức ảnh** vì ở mục này, mô hình đưa ra một nhãn dự đoán cho toàn bộ khung hình/bối cảnh chung của bức ảnh chứ không phát hiện vận thể hay phân loại đối tượng
3. Danh sách tên lớp (class list) do đã được huấn luyện phân loại dựa trên tập các lớp cố định được định nghĩa trước, đi kèm checkpoint `yolo11n-cls.pt` và taxonomy `ImageNet-1K`.
4. Ghi cả 3 vì:
    `class_id`: để đảm bảo tính đồng bộ về mặt định danh kỹ thuật và lập trình.
    `class_name`: giúp reviewer đọc và hiểu ý nghĩa class
    `taxonomy_name`: Định nghĩa ngữ cảnh và không gian tập nhãn được sử dụng. Cùng class_id, class_name nhưng khác taxonomy có thể có ý nghĩa khác nhau và thuộc tính khác nhau
5. Guideline cần quy định rõ tiêu chí ưu tiên chọn chủ đề chính
6. Vì `score` chỉ đại diện cho độ chính xác thuật toán của mô hình dựa trên dữ liệu đã học. Không đảm bảo độ chính xác tuyệt đối, vẫn có thể bị nhầm lẫn. Ground Truth đòi hỏi độ chính xác tuyệt đối do con người gán nhãn theo guideline

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=bus ; `score`=0.912558 ; `bbox_xyxy`=[93.17, 183.95, 223.01, 320.91] ; `bbox_width`=129.84 ; `bbox_height`=132.96
2. Giải thích box bằng lời thường: Từ góc trên-trái (`93.17`, `183.95`) từ góc dưới-phải (`223.01`, `320.91`), chiều rộng bounding box là `129.84px`, chiều cao là `132.96px`
3. Chọn hai ngưỡng (vd. 0.20 và 0.60): Ngưỡng `0.20` có 17 predictions. Ngưỡng `0.60` có 6 predictions
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), nhiều hộp hơn thì **reviewer** phải mất thời gian lọc ra các kết quả nhiễu, không chính xác, lặp lại, đè hộp do mô hình đưa ra nhiều dự đoán có độ tin cậy thấp.
5. **Quy tắc**: "Vẽ khung bao ôm sát toàn bộ các mép ngoài cùng của vật thể, không bỏ sót bộ phận của vật và không để thừa quá nhiều khoảng trống ở các mép/cạnh".
6. Viết thành quy tắc guideline để tất cả annotator áp dụng nhất quán, tránh case nào cũng escalate gây nghẽn tiến độ công việc

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=traffic-002 ; `class_name`=car ; `score`=0.891092 ; số điểm polygon=`polygon_point_count`=92 ; 3–5 điểm đầu của `polygon_xy`=[[497.0, 287.0], [497.0, 287.0], [492.0, 287.0], [491.0, 287.0]]
2. So với box chỉ cho biết phạm vi tối đa hình chữ nhật bao quanh vật thể trong khi polygon cho thêm thông tin chi tiết về đường biên chính xác theo viền thực tế của đối tượng được phát hiện. 
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn chuỗi chữ, id đơn lớp cho toàn bộ ảnh  | Ảnh có nhiều chủ đề khác nhau | Xác định chủ đề chính của ảnh và chọn đúng nhãn theo guideline | Kiểm tra nhãn có tuân thủ quy tắc ưu tiên chủ đề chính trong guideline không |
| Phát hiện vật thể | Tọa độ hình chữ nhật kèm nhãn đối tượng | Model tạo nhiều hộp đè lên nhau hoặc thiếu sót các vật bị che, đè lên hoặc ở mép ảnh | Vẽ bounding box cho vật thể, gán nhãn đúng lớp cho đối tượng | Kiểm tra khung bounding xem có bị vẽ thừa hay thiếu pixel hoặc sót vật thể không |
| Instance segmentation | Tập hợp đa giác bám sát theo viền vật thể, class_name, instance_id | Viền mờ, bóng râm tiếp giáp vật thể, hoặc các vật thể đè lên nhau | Vẽ đường viền polygon bao quanh viền thực tế của từng vật thể | Kiểm tra độ chính xác của đường viền, xem có bị thiếu, thừa hay dính vào vật thể khác không |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của tôi cho bài lab này: Không ghi họ tên, MSSV hoặc bất kỳ dữ liệu nhạy cảm nào trong báo cáo/output
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và escalate bằng cách báo cáo trực tiếp cho Lab Coach ở đó

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
