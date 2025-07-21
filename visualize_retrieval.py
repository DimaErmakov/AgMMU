import json
from collections import defaultdict
import os

# Paths to your files
SAMPLE_JSONL = "/work/nvme/bdbf/dermakov/agmmu/AgMMU_v1/query/test/mbeir_sample_test.jsonl"
# SAMPLE_JSONL = "/u/dermakov/sample.jsonl"
CAND_POOL_JSONL = "/work/nvme/bdbf/dermakov/agmmu/AgMMU_v1/cand_rag3.jsonl"
RUN_FILE = "/u/dermakov/UniIR/retrieval_results/CLIP_SF/Large/Instruct/InBatch/run_files/mbeir_sample_single_pool_test_k10_run.txt"
TOP_K = 3
IMAGES_FT_ROOT = "/work/nvme/bdbf/dermakov/agmmu/AgMMU_v1"


# Helper to get the numeric part after the colon
def get_numeric_id(qid):
    # e.g., '17:379291' -> '379291', '1:8379291' -> '8379291'
    # return qid.split(":")[-1]
    return qid[2:]

# Load queries and build a mapping from numeric id to full qid
numeric_to_qid = {}
qid_to_query = {}
with open(SAMPLE_JSONL, "r") as f:
    for line in f:
        q = json.loads(line)
        qid_to_query[q["qid"]] = q
        numeric_to_qid[get_numeric_id(q["qid"])] = q["qid"]

# Load candidate pool
did_to_cand = {}
with open(CAND_POOL_JSONL, "r") as f:
    for line in f:
        c = json.loads(line)
        did_to_cand[c["did"][2:]] = c

# Parse run file and group by numeric query id
retrieved = defaultdict(list)
with open(RUN_FILE, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        run_qid, _, did, rank, score, *_ = parts
        num_id = get_numeric_id(run_qid)
        if did[1:] in did_to_cand:
            # print(f"num_id: {num_id}")
            retrieved[num_id].append((int(rank), did[1:], float(score)))
# for k, v in retrieved.items():
#     print(f"k: {k}, v: {v}")

# Visualize and save output
results = []
for num_id, qid in numeric_to_qid.items():
    query = qid_to_query[qid]
    query_text = query['query_txt']
    


    query_img = os.path.join(IMAGES_FT_ROOT, query['query_img_path']).replace("./", "")
    top = sorted(retrieved.get(num_id, []), key=lambda x: x[0])[:TOP_K]
    candidates = []
    for rank, did, score in top:
        cand = did_to_cand.get(did)
        if cand:
            candidates.append({
                "rank": rank,
                "did": did,
                "txt": cand['txt'],
                "img": os.path.join(IMAGES_FT_ROOT, cand['img_path']).replace("./", ""),
                "score": score
            })

            # print(f"did: {did}, txt: {cand['txt']}, img: {cand['img_path']}, score: {score}")
    if candidates:
        # Include all MBEIR attributes from the query
        result_entry = {
            "qid": qid[2:],
            "query_text": query_text,
            "query_img": query_img,
            "candidates": candidates,
            # "faq-id": query.get("faq-id"),
            # "title": query.get("title"),
            # "created": query.get("created"),
            # "updated": query.get("updated"),
            # "state": query.get("state"),
            # "county": query.get("county"),
            # "tags": query.get("tags"),
            # "attachments": query.get("attachments"),
            # "question": query.get("question"),
            # "answer": query.get("answer"),
            # "species": query.get("species"),
            # "category": query.get("category"),
            # "qa_information": query.get("qa_information"),
            # "agmmu_question": query.get("agmmu_question"),
            # "qtype": query.get("qtype")
        }
        results.append(result_entry)
        # print("Test")

# Save output to JSON file
with open("retrieval_results.json", "w") as f:
    json.dump(results, f, indent=2)


# # Print all numeric query IDs in sample.jsonl
# print("Numeric IDs in sample.jsonl:")
# print(set(numeric_to_qid.keys()))

# # Print all numeric query IDs in run file
# run_numeric_ids = set()
# with open(RUN_FILE, "r") as f:
#     for line in f:
#         parts = line.strip().split()
#         if len(parts) < 6:
#             continue
#         run_qid = parts[0]
#         run_numeric_ids.add(get_numeric_id(run_qid))
# print("Numeric IDs in run file:")
# print(run_numeric_ids)

# # Print intersection
# print("Intersection:")
# print(set(numeric_to_qid.keys()) & run_numeric_ids)