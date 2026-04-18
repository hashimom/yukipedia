import os
import re
import json
import csv
import argparse
import concurrent.futures
from datasets import load_dataset, Dataset, Features, Value
from huggingface_hub import HfApi
from tqdm import tqdm
from sudachipy import tokenizer
from sudachipy import dictionary

POS_ID_FILE = "pos_id.csv"
STATE_FILE = "resume_state.json"
sudachi_tokenizer = dictionary.Dictionary(dict="full").create()


POS_ID_DEF = []
with open(POS_ID_FILE, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        POS_ID_DEF.append(row[0] + "." + row[1] + "." + row[2] + "." + row[3] + "." + row[4] + "." + row[5])


# ==========================================
# 1. スキーマ（データ構造）の厳密な定義
# ==========================================
# DatasetBuilderの代わりにここで型を定義します
CUSTOM_FEATURES = Features({
    "id": Value("string"),
    "text": Value("string"),
})


# ==========================================
# 2. 状態管理（レジューム）用の関数
# ==========================================
def load_state():
    """前回の保存状態があれば読み込む"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"file_index": 0, "total_consumed": 0, "chunk_data": []}


def save_state(file_index, total_consumed, chunk_data):
    temp_file = STATE_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump({
            "file_index": file_index,
            "total_consumed": total_consumed,
            "chunk_data": chunk_data
        }, f, ensure_ascii=False)
    os.replace(temp_file, STATE_FILE)


# ==========================================
# 3. マルチプロセス用の加工関数（トップレベルに配置）
# ==========================================
def filter_no_alphanumeric(text_list):
    pattern = re.compile(r'[a-zA-Z0-9]')
    return [text for text in text_list if not pattern.search(text)]

def process_batch(batch):
    processed_data = []
    for example in batch:
        samples = re.split('[。？ ]', example["text"])[:-1]
        samples = filter_no_alphanumeric(samples)
        for sample in samples:
            token = sudachi_tokenizer.tokenize(sample, tokenizer.Tokenizer.SplitMode.C)
            if 4 < len(token) < 128:
                m_str = ""
                for m in token:
                    pos_ary = m.part_of_speech()
                    pos = pos_ary[0] + "." + pos_ary[1] + "." + pos_ary[2] + "." + pos_ary[3] + "." + pos_ary[4] + "." + pos_ary[5]
                    m_str += m.surface() + "/" + str(POS_ID_DEF.index(pos)) + " "
                processed_data.append({"id": example["id"], "text": m_str.strip()})
    return processed_data, len(batch)


# ==========================================
# 4. データをバッチにまとめるジェネレーター
# ==========================================
def get_batches(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def upload_task(file_name, repo_id, api):
    api.upload_file(
        path_or_fileobj=file_name,
        path_in_repo=f"data/{file_name}",
        repo_id=repo_id,
        repo_type="dataset"
    )
    os.remove(file_name)
    return file_name


# ==========================================
# 5. メイン処理
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='yuinotrain')
    parser.add_argument('-r', '--repo_id', default="hashimom/yukipedia")
    parser.add_argument('--data_cache_dir', default="~/hf_datasets", help="data cache path")
    parser.add_argument('--chunk_size', type=int, default=20000000)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--num_processes', type=int, default=32)
    args = parser.parse_args()

    max_queue_tasks = args.num_processes * 2
    max_concurrent_uploads = 1

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)

    # 状態の復元
    state = load_state()
    file_index = state["file_index"]
    total_consumed = state["total_consumed"]
    chunk_data = state["chunk_data"]

    print("CC100のストリーミングを開始します...")
    dataset = load_dataset("range3/cc100-ja", cache_dir=args.data_cache_dir)
    stream_data = dataset["train"]

    if total_consumed > 0:
        print(f"🔄 前回の状態を復元しました。")
        print(f" - {total_consumed} 件目から読み込みを再開します。")
        print(f" - アップロードは data_{file_index:05d}.parquet から再開します。")
        stream_data = stream_data.skip(total_consumed)

    upload_futures = {}
    with (concurrent.futures.ProcessPoolExecutor(max_workers=args.num_processes) as executor,
          concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_uploads) as upload_executor):
        batch_iterator = get_batches(stream_data, args.batch_size)
        process_futures = set()

        with tqdm(desc="読込・加工進捗", unit="件", initial=total_consumed) as pbar:
            while True:
                while len(process_futures) < max_queue_tasks:
                    try:
                        batch = next(batch_iterator)
                        process_futures.add(executor.submit(process_batch, batch))
                    except StopIteration:
                        break

                if not process_futures and not upload_futures:
                    break

                if not process_futures:
                    # 全データの読み込みが終わり、アップロードの完了だけを待つ状態
                    concurrent.futures.wait(
                        upload_futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED
                    )
                else:
                    # どれか1つの「加工タスク」が完了するまで待機
                    done_processes, process_futures = concurrent.futures.wait(
                        process_futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    for future in done_processes:
                        processed_data, original_count = future.result()
                        chunk_data.extend(processed_data)
                        total_consumed += original_count
                        pbar.update(original_count)

                        if len(chunk_data) >= args.chunk_size:
                            upload_data = chunk_data[:args.chunk_size]
                            chunk_data = chunk_data[args.chunk_size:]

                            file_name = f"data_{file_index:05d}.parquet"
                            temp_dataset = Dataset.from_list(upload_data, features=CUSTOM_FEATURES)
                            temp_dataset.to_parquet(file_name)

                            up_future = upload_executor.submit(upload_task, file_name, args.repo_id, api)
                            upload_futures[up_future] = {
                                "file_index": file_index + 1,
                                "total_consumed": total_consumed,
                                "chunk_data": chunk_data.copy()
                            }
                            file_index += 1

                done_uploads = [f for f in list(upload_futures.keys()) if f.done()]
                if done_uploads:
                    latest_state = None
                    max_idx = -1

                    for f in done_uploads:
                        try:
                            f.result() # ここでエラーが起きていれば例外が発生する
                            state_info = upload_futures[f]

                            if state_info["file_index"] > max_idx:
                                max_idx = state_info["file_index"]
                                latest_state = state_info
                        except Exception as e:
                            tqdm.write(f"❌ アップロード失敗: {e}")
                        finally:
                            del upload_futures[f]

                    if latest_state:
                        save_state(
                            latest_state["file_index"],
                            latest_state["total_consumed"],
                            latest_state["chunk_data"]
                        )

    # ==========================================
    # 6. 全件完了後の端数アップロード
    # ==========================================
    if len(chunk_data) > 0:
        file_name = f"data_{file_index:05d}.parquet"
        temp_dataset = Dataset.from_list(chunk_data, features=CUSTOM_FEATURES)
        temp_dataset.to_parquet(file_name)
        tqdm.write(f"最後のファイル {file_name} をアップロード中...")
        api.upload_file(
            path_or_fileobj=file_name,
            path_in_repo=f"data/{file_name}",
            repo_id=args.repo_id,
            repo_type="dataset"
        )
        os.remove(file_name)
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)

    api.upload_file(
        path_or_fileobj=POS_ID_FILE,
        path_in_repo=POS_ID_FILE,
        repo_id=args.repo_id,
        repo_type="dataset"
    )
    print("🎉 すべての処理とアップロードが完了しました！")


if __name__ == "__main__":
    main()
