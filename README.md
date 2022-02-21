# sahi-yolox-onnx-sample
[SAHI(Slicing Aided Hyper Inference)](https://github.com/obss/sahi)を[YOLOX(ONNX)](https://github.com/Kazuhito00/YOLOX-ONNX-TFLite-Sample)で動かしたサンプルです。<bR>
  
https://user-images.githubusercontent.com/37477845/154991823-f1b6297f-fd00-48f9-a59f-604e2c7a526a.mp4
  
  
左図：通常推論(sample_prediction.py)　右図：SAHI(sample_sliced_prediction.py)<br>

# Requirement 
* onnxruntime 1.10.0 or later
* Shapely 1.8.1 or later

# Demo
デモの実行方法は以下です。
```bash
python sample_sliced_prediction.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --model<br>
ロードするYOLOXモデルの格納パス<br>
デフォルト：yolox/model/yolox_nano.onnx
* --config<br>
YOLOXのコンフィグファイル格納パス<br>
デフォルト：yolox/config.json
* --slice_height<br>
SAHIの画像スライス高さ<br>
デフォルト：512
* --slice_width<br>
SAHIの画像スライス幅<br>
デフォルト：512
* --overlap_height_ratio<br>
SAHIの画像スライス時の高さ方向のオーバーラップ率<br>
デフォルト：0.2
* --overlap_width_ratio<br>
SAHIの画像スライス時の幅方向のオーバーラップ率<br>
デフォルト：0.2
* --draw_score_th<br>
描画時のスコア閾値<br>
デフォルト：0.3

# Reference
* [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [obss/sahi](https://github.com/obss/sahi)
* [Kazuhito00/YOLOX-ONNX-TFLite-Sample](https://github.com/Kazuhito00/YOLOX-ONNX-TFLite-Sample)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
sahi-yolox-onnx-sample is under [MIT License](LICENSE).

# License(Movie)
サンプル動画は[NHKクリエイティブ・ライブラリー](https://www.nhk.or.jp/archives/creative/)の[カモメのハンティング](https://www2.nhk.or.jp/archives/creative/material/view.cgi?m=D0002030249_00000)を使用しています。
