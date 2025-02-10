import json

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            new_line = json.loads(line.strip())
            new_line_answer = new_line["answer"].lower()
            if new_line_answer == "yes" or new_line_answer == "no" or new_line_answer == "insufficient information.":
                continue
            data.append(new_line)
            
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for line in data:
            file.write(json.dumps(line, ensure_ascii=False) + '\n')
        
    # import pdb; pdb.set_trace()
    return data

# 示例用法
file_path = '/home/yaodong/codes/GNNRAG/IdsData/multihop-rag/Question.json'
output_file_path = '/home/yaodong/codes/GNNRAG/IdsData/multihop-rag/Filtered_Question.json'
data = read_jsonl(file_path)
print(data)