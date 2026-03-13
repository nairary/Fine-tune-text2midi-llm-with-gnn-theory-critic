import argparse
import json

def count_tracks(path: str):
    unique_ids = set()

    with open(path, 'r') as file:
        for line in file:
            json_line = json.loads(line)
            unique_ids.add(json_line.get("ori_uid", ""))

    print("="*8)
    print(f"{path}")
    print(f"Unique oids: {len(unique_ids)}")


def pasrse_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument("--structure-train", type=str, required=True, help="Path to HookTheoryStructure.train.jsonl")
    parser.add_argument("--structure-val", type=str, required=True, help="Path to HookTheoryStructure.val.jsonl")
    parser.add_argument("--structure-test", type=str, required=True, help="Path to HookTheoryStructure.test.jsonl")

    return parser.parse_args()

def main():
    args = pasrse_cmd()

    train_path = args.structure_train
    test_path = args.structure_test
    val_path = args.structure_val

    for path in [train_path, test_path, val_path]:
        count_tracks(path)

if __name__ == "__main__":
    main()
