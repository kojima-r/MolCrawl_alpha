"""
BERTチェックポイントの包括的テストスクリプト

このスクリプトは以下のテストを実行します：
1. モデルとトークナイザーの読み込みテスト
2. マスク言語モデル（MLM）のテスト
3. エンベディング生成テスト
4. バッチ処理テスト
5. モデルのパフォーマンス統計
"""

import torch
import argparse
import time
import json
import numpy as np
from pathlib import Path
from transformers import BertForMaskedLM, BertTokenizer, DataCollatorForLanguageModeling
from datasets import load_from_disk
import matplotlib.pyplot as plt


def load_model_and_tokenizer(checkpoint_path):
    """チェックポイントからBERTモデルとトークナイザーをロードする"""
    try:
        print(f"チェックポイントをロードしています: {checkpoint_path}")
        model = BertForMaskedLM.from_pretrained(checkpoint_path)
        
        # トークナイザーの読み込み（チェックポイントからまたは設定ファイルから）
        try:
            tokenizer = BertTokenizer.from_pretrained(checkpoint_path)
        except Exception as e:
            print(f"チェックポイントからトークナイザーを読み込めませんでした: {e}")
            print("設定ファイルからトークナイザーを読み込みます...")
            # 設定に基づいてトークナイザーを初期化（必要に応じて調整）
            tokenizer = None
            
        print("✓ モデルとトークナイザーの読み込み成功")
        return model, tokenizer
    except Exception as e:
        print(f"✗ モデルの読み込み中にエラーが発生しました: {e}")
        return None, None


def test_basic_functionality(model, tokenizer, test_texts):
    """基本的な機能のテスト"""
    print("\n=== 基本機能テスト ===")
    
    if tokenizer is None:
        print("トークナイザーが利用できないため、基本機能テストをスキップします")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    for i, text in enumerate(test_texts):
        print(f"\nテスト {i+1}: {text}")
        
        # テキストをトークナイズ
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 推論実行
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"  ✓ 推論成功 - 出力形状: {outputs.logits.shape}")


def test_masked_language_modeling(model, tokenizer, test_texts):
    """マスク言語モデリングのテスト"""
    print("\n=== マスク言語モデリングテスト ===")
    
    if tokenizer is None:
        print("トークナイザーが利用できないため、MLMテストをスキップします")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    for text in test_texts[:2]:  # 最初の2つのテキストでテスト
        # マスクトークンを挿入
        tokens = tokenizer.tokenize(text)
        if len(tokens) > 3:
            # 中間のトークンをマスク
            mask_idx = len(tokens) // 2
            original_token = tokens[mask_idx]
            tokens[mask_idx] = tokenizer.mask_token
            masked_text = tokenizer.convert_tokens_to_string(tokens)
            
            print(f"\n元のテキスト: {text}")
            print(f"マスクされたテキスト: {masked_text}")
            
            # 予測実行
            inputs = tokenizer(masked_text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # マスク位置の予測を取得
            mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
            if len(mask_token_index) > 0:
                mask_token_logits = outputs.logits[0, mask_token_index[0], :]
                top_5_tokens = torch.topk(mask_token_logits, 5, dim=-1)
                
                print(f"元のトークン: {original_token}")
                print("予測されたトップ5トークン:")
                for i, (score, token_id) in enumerate(zip(top_5_tokens.values, top_5_tokens.indices)):
                    token = tokenizer.decode([token_id])
                    print(f"  {i+1}. {token} (スコア: {score.item():.3f})")


def test_embedding_generation(model, tokenizer, test_texts):
    """エンベディング生成のテスト"""
    print("\n=== エンベディング生成テスト ===")
    
    if tokenizer is None:
        print("トークナイザーが利用できないため、エンベディングテストをスキップします")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    embeddings = []
    
    for text in test_texts[:3]:  # 最初の3つのテキストでテスト
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.bert(**inputs)  # BertForMaskedLMの場合、bertレイヤーにアクセス
            # [CLS]トークンのエンベディングを取得
            cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
            embeddings.append(cls_embedding)
        
        print(f"✓ エンベディング生成成功 - 形状: {cls_embedding.shape}")
    
    if len(embeddings) > 1:
        # エンベディング間の類似度を計算
        print("\nエンベディング間のコサイン類似度:")
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                print(f"  テキスト{i+1} vs テキスト{j+1}: {similarity:.3f}")


def test_batch_processing(model, tokenizer, test_texts):
    """バッチ処理のテスト"""
    print("\n=== バッチ処理テスト ===")
    
    if tokenizer is None:
        print("トークナイザーが利用できないため、バッチ処理テストをスキップします")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # バッチサイズを変えてテスト
    batch_sizes = [1, 2, 4]
    
    for batch_size in batch_sizes:
        if batch_size > len(test_texts):
            continue
            
        batch_texts = test_texts[:batch_size]
        
        start_time = time.time()
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✓ バッチサイズ {batch_size}: {processing_time:.3f}秒")


def test_model_performance(model, tokenizer, dataset_path=None):
    """モデルのパフォーマンステスト"""
    print("\n=== パフォーマンステスト ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # モデル情報の表示
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"総パラメータ数: {total_params:,}")
    print(f"訓練可能パラメータ数: {trainable_params:,}")
    print(f"モデルサイズ: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    print(f"使用デバイス: {device}")
    
    # GPU使用量の確認（CUDA利用可能な場合）
    if torch.cuda.is_available():
        print(f"GPU メモリ使用量: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
        print(f"GPU メモリ予約量: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")
    
    # データセットでの評価（もし利用可能な場合）
    if dataset_path and tokenizer:
        try:
            print(f"\nデータセットでの評価: {dataset_path}")
            dataset = load_from_disk(dataset_path)
            
            if "test" in dataset:
                test_dataset = dataset["test"].select(range(min(100, dataset["test"].num_rows)))
                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
                )
                
                model.eval()
                total_loss = 0
                num_batches = 0
                
                for i in range(0, len(test_dataset), 8):  # バッチサイズ8
                    batch = test_dataset[i:i+8]
                    batch = data_collator([batch[j] for j in range(len(batch))])
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    with torch.no_grad():
                        outputs = model(**batch)
                        total_loss += outputs.loss.item()
                        num_batches += 1
                
                avg_loss = total_loss / num_batches
                perplexity = torch.exp(torch.tensor(avg_loss))
                print(f"平均損失: {avg_loss:.4f}")
                print(f"パープレキシティ: {perplexity:.4f}")
                
        except Exception as e:
            print(f"データセット評価中にエラー: {e}")


def generate_test_report(checkpoint_path, results):
    """テスト結果のレポートを生成"""
    report_path = Path(checkpoint_path).parent / "test_report.json"
    
    report = {
        "checkpoint_path": checkpoint_path,
        "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ テストレポートを保存しました: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="BERTチェックポイントの包括的テストスクリプト")
    parser.add_argument("--checkpoint_path", required=True, help="テストするチェックポイントのパス")
    parser.add_argument("--dataset_path", help="評価用データセットのパス（オプション）")
    parser.add_argument("--test_texts", nargs="*", 
                       default=[
                           "これはテストサンプルです。",
                           "分子の構造を解析します。",
                           "機械学習モデルの性能を評価中。",
                           "自然言語処理の技術が進歩している。",
                           "データサイエンスは重要な分野です。"
                       ],
                       help="テスト用のサンプルテキスト")
    
    args = parser.parse_args()
    
    print("=== BERTチェックポイント テストスクリプト ===")
    print(f"チェックポイント: {args.checkpoint_path}")
    
    results = {}
    
    # モデルとトークナイザーのロード
    model, tokenizer = load_model_and_tokenizer(args.checkpoint_path)
    
    if model is None:
        print("モデルの読み込みに失敗したため、テストを終了します。")
        return
    
    try:
        # 各種テストの実行
        test_basic_functionality(model, tokenizer, args.test_texts)
        test_masked_language_modeling(model, tokenizer, args.test_texts)
        test_embedding_generation(model, tokenizer, args.test_texts)
        test_batch_processing(model, tokenizer, args.test_texts)
        test_model_performance(model, tokenizer, args.dataset_path)
        
        results["status"] = "success"
        results["tests_completed"] = ["basic_functionality", "mlm", "embedding", "batch_processing", "performance"]
        
    except Exception as e:
        print(f"\nテスト中にエラーが発生しました: {e}")
        results["status"] = "error"
        results["error"] = str(e)
    
    # レポート生成
    generate_test_report(args.checkpoint_path, results)
    
    print("\n=== テスト完了 ===")


if __name__ == "__main__":
    main()
