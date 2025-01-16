import os
import logging
import pandas as pd
from tqdm import tqdm

# エラーハンドリングとモジュールインポート
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        pipeline,
        AdamW,
        get_scheduler
    )
    import torch
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import accuracy_score
except ModuleNotFoundError as e:
    raise ImportError(f"Required module not found: {e}. Please install dependencies using 'pip install -r requirements.txt'")

# ロギング設定
logging.basicConfig(
    filename='sentiment_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SentimentStanceAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # 感情分析用モデル
        self.sentiment_model_name = "koheiduck/bert-japanese-finetuned-sentiment"
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_name)
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=self.sentiment_model_name,
            tokenizer=self.sentiment_tokenizer
        )
        
        # スタンス分析用モデル
        self.stance_model_name = "cl-tohoku/bert-base-japanese"
        self.stance_tokenizer = AutoTokenizer.from_pretrained(self.stance_model_name)
        self.stance_model = AutoModelForSequenceClassification.from_pretrained(
            self.stance_model_name, 
            num_labels=5  # 5クラス分類 (強く賛成/賛成/中立/反対/強く反対)
        )
        self.stance_model.to(self.device)

    def analyze_sentiment(self, text):
        """テキストの感情分析を行う"""
        try:
            result = self.sentiment_analyzer(text)
            label = result[0]['label']
            score = result[0]['score']
            
            # 動的閾値計算
            text_length = len(text)
            base_threshold = 0.50
            length_factor = max(0.5, min(1.0, text_length / 20))  # 最小0.5を保証
            dynamic_threshold = base_threshold + (0.03 * length_factor)  # 係数を0.03に調整
            
            # 感情分類（3段階）
            if label == 'POSITIVE':
                label = 'POSITIVE'
            elif label == 'NEGATIVE':
                label = 'NEGATIVE'
            else:
                label = 'NEUTRAL'
                
            # スコア正規化（閾値以下でも相対スコアを保持）
            if score > dynamic_threshold:
                normalized_score = (score - dynamic_threshold) / (1.0 - dynamic_threshold)
            else:
                normalized_score = (score - 0.0) / dynamic_threshold  # 0.0から閾値までの相対スコア
            normalized_score = max(0.0, min(1.0, normalized_score))
                
            # 小数第4位を四捨五入
            rounded_score = round(normalized_score, 3)
            return {
                'text': text,
                'sentiment': f"{label} ({rounded_score:.3f})",
                'score': rounded_score
            }
        except Exception as e:
            logging.error(f"Sentiment analysis error: {e}")
            return None

    def analyze_stance(self, text):
        """テキストのスタンス分析を行う"""
        try:
            encoding = self.stance_tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.stance_model(**encoding)
            
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
            prediction = torch.argmax(logits, dim=-1).cpu().numpy()[0]
            # 小数第4位を四捨五入
            rounded_score = round(float(probs[prediction]), 3)
            
            # 5段階評価に変換
            if prediction == 0:
                stance_label = "強く賛成"
            elif prediction == 1:
                stance_label = "賛成"
            elif prediction == 2:
                stance_label = "中立"
            elif prediction == 3:
                stance_label = "反対"
            else:
                stance_label = "強く反対"
                
            return {
                'text': text,
                'stance': f"{stance_label} ({rounded_score:.3f})",
                'stance_score': rounded_score
            }
        except Exception as e:
            logging.error(f"Stance analysis error: {e}")
            return None

    def analyze_batch(self, texts):
        """バッチ処理による感情・スタンス分析"""
        results = []
        for i, text in enumerate(tqdm(texts, desc="Analyzing texts")):
            sentiment_result = self.analyze_sentiment(text)
            stance_result = self.analyze_stance(text)
            
            if sentiment_result and stance_result:
                results.append({
                    'number': i + 1,
                    'text': text,
                    'sentiment': sentiment_result['sentiment'],
                    'sentiment_score': sentiment_result['score'],
                    'stance': stance_result['stance'],
                    'stance_score': stance_result['stance_score']
                })
        
        return pd.DataFrame(results)

    def train_stance_model(self, train_texts, train_labels, val_texts, val_labels, num_epochs=3):
        """スタンス分析モデルのファインチューニング"""
        class StanceDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length=128):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = self.texts[idx]
                label = self.labels[idx]
                encoding = self.tokenizer(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(label, dtype=torch.long)
                }

        train_dataset = StanceDataset(train_texts, train_labels, self.stance_tokenizer)
        val_dataset = StanceDataset(val_texts, val_labels, self.stance_tokenizer)

        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=8)

        optimizer = AdamW(self.stance_model.parameters(), lr=5e-5)
        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear", 
            optimizer=optimizer, 
            num_warmup_steps=0, 
            num_training_steps=num_training_steps
        )

        self.stance_model.train()
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.stance_model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        # モデル評価
        self.stance_model.eval()
        predictions = []
        true_labels = []

        for batch in val_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.stance_model(**batch)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())

        accuracy = accuracy_score(true_labels, predictions)
        logging.info(f"Stance model validation accuracy: {accuracy}")
        return accuracy

def analyze_csv(input_path, output_path, text_column=None, batch_size=100):
    """CSVファイルを分析する"""
    try:
        # CSV読み込み（ヘッダーあり）
        df = pd.read_csv(input_path, header=0)
        
        # テキストカラムの指定（'内容'列を使用）
        text_column = '内容' if '内容' in df.columns else 0
        texts = df[text_column].tolist()  # 全ての行を分析
        
        # バッチ処理
        analyzer = SentimentStanceAnalyzer()
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results = analyzer.analyze_batch(batch)
            all_results.append(results)
            print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} comments")
            
        # 結果を結合して保存
        final_df = pd.concat(all_results)
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logging.info(f"Analysis completed. Results saved to {output_path}")
        
    except Exception as e:
        logging.error(f"CSV analysis error: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='analysis_output.csv',
                       help='Path to save output CSV file')
    parser.add_argument('--column', type=int, default=None,
                       help='Text column index (default: 0)')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Number of comments per batch')
    
    args = parser.parse_args()
    
    # 分析実行
    analyze_csv(
        input_path=args.input,
        output_path=args.output,
        text_column=args.column,
        batch_size=args.batch_size
    )
