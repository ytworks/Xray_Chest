# AIエンジンの使い方
## ディレクトリ構成
* Config
  * 推論時のパラメータ設定ファイル置き場
* Data
  * 学習用データ置き場。必要なデータは下記
  * https://nihcc.app.box.com/v/ChestXray-NIHCC
  * http://imgcom.jsrt.or.jp/download/
* DICOMReader
  * DICOM読み込み用モジュール
* LinearMotor
  * tensorflow用モジュール
* Pic
  * 出力画像置き場
* preprocessing_tool
  * 前処理用モジュール
* Result
  * 結果ファイル置き場
* scripts
  * 実行ファイルスクリプト群
* setting
  * パラメータ設定ファイル群
* detection.py
  * ニューラルネットワークモデル
* main_prediction.py
  * 推論実行モデル
* main.py
  * 学習実行モデル
* make_mini_batch.py
  * ミニバッチ作成モジュール
* NetworkModules.py
  * ニューラルネットワークモデルライブラリ
* SimpleCells.py
  * ニューラルネットワークモデルライブラリ
* util.py
  * 共通関数
* write_roc.py
  * ROCカーブ出力モジュール
* README.md
  * これ
* requirements.txt
  * pipファイル


## 実行ファイル群
* run_dev.sh
  * 開発用実行ファイル。本番よりも軽くて簡単なモデルを実行
  * setting/dev.iniでパラメータ設定可能
  * -d debugで学習時の最後に実行されるAWSインスタンスを落とすスクリプトを実行しないことができる
  * -v で学習時に使っているAWSのanaconda仮想環境の移動を実行しないことができる
* run_local.sh
  * ローカル用実行ファイル。本番よりも軽くて簡単なモデルを実行
  * setting/local.iniでパラメータ設定可能
  * -d debugで学習時の最後に実行されるAWSインスタンスを落とすスクリプトを実行しないことができる
  * -v で学習時に使っているAWSのanaconda仮想環境の移動を実行しないことができる
* run_prediction.sh
  * 本番用推論実行ファイル。
  * setting/prediction.iniでパラメータ設定可能
  * -i インプット画像データファイルのファイルパスを指定
  * -o 結果画像の出力先ディレクトリを指定
  * -d debugで学習時の最後に実行されるAWSインスタンスを落とすスクリプトを実行しないことができる
  * -v で学習時に使っているAWSのanaconda仮想環境の移動を実行しないことができる
* run_prod.sh
  * 本番用学習実行ファイル。テストのROI画像は生成しない
  * setting/prod.iniでパラメータ設定可能
  * -d debugで学習時の最後に実行されるAWSインスタンスを落とすスクリプトを実行しないことができる
  * -v で学習時に使っているAWSのanaconda仮想環境の移動を実行しないことができる
* run_roi.sh
  * 本番用学習実行ファイル。テストのROI画像を生成する
  * setting/prod_roi.iniでパラメータ設定可能
  * -d debugで学習時の最後に実行されるAWSインスタンスを落とすスクリプトを実行しないことができる
  * -v で学習時に使っているAWSのanaconda仮想環境の移動を実行しないことができる
