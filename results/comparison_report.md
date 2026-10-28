# Road Damage Detection – Model Comparison Report

## Dataset
| Property | Value |
|----------|-------|
| Images   | 6,757 (4930 train / 1146 val / 681 test) |
| Classes  | 7 (alligator, block, crack, edge, longitudinal, pothole, transverse) |
| Format   | YOLOv8 (Roboflow) |

## Metrics Comparison

| Model | mAP@50 | mAP@50-95 | Precision | Recall | FPS (CPU) | ms/img | Size (MB) |
|-------|--------|-----------|-----------|--------|-----------|--------|-----------|
| YOLO26n_default ⭐ | 0.0001 | 0.0000 | 0.0009 | 0.0357 | 1.8 | 566.9 | 5.2 |
| YOLOv11n_default | 0.0000 | 0.0000 | 0.0001 | 0.0174 | 0.7 | 1350.7 | 5.2 |
| YOLOv8n_default | 0.0000 | 0.0000 | 0.0005 | 0.0091 | 0.8 | 1297.0 | 6.2 |

## Best Model
**YOLO26n_default** achieved the highest mAP@50 of **0.0001**.

## Analysis
- Models are compared across detection accuracy (mAP), efficiency (FPS, size), and deployment suitability.
- For embedded/edge deployment, models with higher FPS and smaller size are preferred.
- Hyperparameter tuning variants show the impact of learning rate and weight decay on convergence.

## Plots
- `comparison_bar_chart.png` – Side-by-side metric bars
- `comparison_radar.png` – Multi-dimensional radar chart
- `speed_vs_accuracy.png` – FPS vs mAP scatter
- `model_size_vs_map.png` – Size vs mAP scatter