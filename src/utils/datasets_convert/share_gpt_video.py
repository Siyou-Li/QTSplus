import json
import os

def convert_jsonl_video(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            conv_id = data["id"]
            video_path = data["video"]
            conversations = data["conversations"]

            video_path = os.path.join(
                "datasets/ShareGPTVideo/train_video_and_instruction/train_300k",
                video_path
            )
            # Process conversation pairs
            for i in range(0, len(conversations), 2):
                human_turn = conversations[i]
                gpt_turn = conversations[i + 1] if i + 1 < len(conversations) else {"value": ""}

                new_entry = {
                    "id": f"{conv_id}",
                    "vison_path": video_path,
                    "data_type": "video",
                    "input": "<|vision|>\n" + human_turn["value"].replace("\n<video>", "").replace("<video>\n", ""),
                    "output": gpt_turn["value"]
                }

                outfile.write(json.dumps(new_entry, ensure_ascii=False) + "\n")

def convert_jsonl_image(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            conv_id = data["id"]
            image_path = data["image"]
            conversations = data["conversations"]

            # Process conversation pairs
            for i in range(0, len(conversations), 2):
                human_turn = conversations[i]
                gpt_turn = conversations[i + 1] if i + 1 < len(conversations) else {"value": ""}

                new_entry = {
                    "id": f"{conv_id}",
                    "vison_path": image_path,
                    "data_type": "image",
                    "input": human_turn["value"],
                    "output": gpt_turn["value"]
                }

                outfile.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
# Example usage
convert_jsonl_video(
    "datasets/ShareGPTVideo/test_video_and_instruction/video_instruction/test/actnet.qa.jsonl", 
    "datasets/fused_dataset/test/ShareGPTVideo/actnet.qa.jsonl"
    )
# convert_jsonl_image(
#     "datasets/ShareGPTVideo/train_video_and_instruction/video_instruction/train/sft/image_instruction_600k.jsonl", 
#     "datasets/fused_dataset/train/ShareGPTVideo/image_instruction_600k.jsonl"
#     )