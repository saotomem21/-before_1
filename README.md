# 感情分析とスタンス分析

このリポジトリは、事前学習済みBERTモデルを使用して日本語テキストの感情分析とスタンス分析を行うPythonスクリプトを含んでいます。

## 特徴
- koheiduck/bert-japanese-finetuned-sentimentモデルを使用した感情分析
- URL削除や空白正規化を含むテキスト前処理
- 大規模データセットの効率的な分析のためのバッチ処理
- 監視とデバッグのための詳細なロギング

## 必要環境
- Python 3.8+
- 依存関係はrequirements.txtを参照

## 使用方法
1. リポジトリをクローン
2. 依存関係をインストール: `pip install -r requirements.txt`
3. 環境変数を設定:
   ```bash
   export HUGGINGFACE_TOKEN=your_token_here
   ```
4. 分析を実行:
   ```bash
   python sentiment_stance_analysis.py
   ```

## 出力
- 分析結果はanalysis_output.csvに保存
- ログはsentiment_analysis.logに保存

## ライセンス
MIT
