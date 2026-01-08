# PP-DocLayoutV2 OpenVINO Inference

OpenVINO-based inference implementation for PP-DocLayoutV2 document layout detection.

## 📋 Introduction

This project provides a complete implementation for running PP-DocLayoutV2 document layout detection model using OpenVINO inference engine. PP-DocLayoutV2 is a document layout analysis model provided by PaddleOCR, capable of detecting various layout elements in documents, such as titles, paragraphs, tables, images, etc.

## ✨ Features

- ✅ Complete OpenVINO IR model inference implementation
- ✅ Fully consistent with PaddleOCR preprocessing and post-processing logic
- ✅ Support for multiple inference devices: CPU, GPU, AUTO
- ✅ Command-line parameterization for easy integration and use
- ✅ Automatic saving of detection results (JSON and visualization images)
- ✅ Support for 25 types of document layout element detection

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 📥 Model Download

The OpenVINO IR model files can be downloaded from ModelScope:

**Model Location:** [PP-DocLayoutV2-ov on ModelScope](https://www.modelscope.cn/models/zhaohb/PP-DocLayoutV2-ov/summary)

After downloading, get model files (`.xml` and `.bin`) and specify the path to the `.xml` file using the `--model_path` parameter.

## 🚀 Quick Start

### Basic Usage

```bash
python ov_infer.py \
    --model_path pp_doclayoutv2.xml \
    --image_path layout.jpg \
    --output_dir ./output
```

### Complete Parameter Example

```bash
python ov_infer.py \
    --model_path pp_doclayoutv2.xml \
    --image_path layout.jpg \
    --output_dir ./output_ov \
    --device GPU \
    --threshold 0.5
```

## 📖 Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `--model_path` | str | ✅ | - | OpenVINO IR model path (.xml file) |
| `--image_path` | str | ✅ | - | Input image path |
| `--output_dir` | str | ❌ | `./output_ov` | Output directory |
| `--device` | str | ❌ | `GPU` | Inference device: `CPU`, `GPU`, `AUTO` |
| `--threshold` | float | ❌ | `0.5` | Detection confidence threshold |

## 📊 Supported Layout Elements

The model can detect the following 25 types of document layout elements:

- `abstract` - Abstract
- `algorithm` - Algorithm
- `aside_text` - Aside text
- `chart` - Chart
- `content` - Content
- `display_formula` - Display formula
- `doc_title` - Document title
- `figure_title` - Figure title
- `footer` - Footer
- `footer_image` - Footer image
- `footnote` - Footnote
- `formula_number` - Formula number
- `header` - Header
- `header_image` - Header image
- `image` - Image
- `inline_formula` - Inline formula
- `number` - Number
- `paragraph_title` - Paragraph title
- `reference` - Reference
- `reference_content` - Reference content
- `seal` - Seal
- `table` - Table
- `text` - Text
- `vertical_text` - Vertical text
- `vision_footnote` - Vision footnote

## 📁 Output Format

### JSON Output

Detection results are saved as JSON files in the following format:

```json
{
  "res": {
    "input_path": "layout.jpg",
    "page_index": null,
    "boxes": [
      {
        "cls_id": 0,
        "label": "text",
        "score": 0.95,
        "coordinate": [xmin, ymin, xmax, ymax]
      },
      ...
    ]
  }
}
```

### Visualization Image

Detection results are saved as visualization images, including:
- Detection box drawing
- Category labels and confidence scores
- Different colors to distinguish different categories

## 🔍 Comparison with PaddleOCR

This project provides `paddle_infer.py` as a reference implementation using PaddleOCR official API for inference. The implementation of `ov_infer.py` is fully consistent with PaddleOCR's preprocessing and post-processing logic, ensuring consistency of inference results.

## 📝 Code Examples

### Python API Usage

```python
from ov_infer import paddle_ov_doclayout

result = paddle_ov_doclayout(
    model_path="pp_doclayoutv2.xml",
    image_path="layout.jpg",
    output_dir="./output",
    device="GPU",
    threshold=0.5
)

print(f"Detected {len(result.boxes)} layout elements")
for box in result.boxes:
    print(f"{box['label']}: {box['score']:.3f} at {box['coordinate']}")
```

## 📄 License

This project is based on PaddleOCR's PP-DocLayoutV2 model. Please follow the corresponding open source license.

## 📧 Contact

For questions or suggestions, please submit an Issue or Pull Request.
