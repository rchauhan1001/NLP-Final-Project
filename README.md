# NLP-Final-Project

This is a final project for the Natural Language Processing class at Northeastern University, Fall 2025

This project implements a fact verification system for the HoVer dataset using BM25 retrieval via Elasticsearch and neural verification models.

## Overview

The system performs multi-hop fact verification in three stages:
1. **Document Retrieval**: BM25-based retrieval using Elasticsearch to find relevant Wikipedia articles
2. **Sentence Selection**: Extract specific sentences from retrieved documents
3. **Claim Verification**: Classify claims as SUPPORTED or NOT_SUPPORTED

## Project Structure
Project/
├── hover_project.py              # Main implementation (WikipediaIndexer, BM25Retriever)
├── run_indexing.py               # Script to index Wikipedia into Elasticsearch
├── run_retrieval.py              # Script to retrieve documents for HoVer claims
├── data/
│   ├── hover_train_release_v1.1.json
│   ├── hover_dev_release_v1.1.json
│   ├── hover_test_release_v1.1.json
│   └── enwiki-20171001-pages-meta-current-withlinks-processed/
│       ├── AA/wiki_00.bz2
│       ├── AB/wiki_00.bz2
│       └── ... (to FZ/)
├── output/
│   ├── hover_train_bm25_top100.json
│   ├── hover_dev_bm25_top100.json
│   └── hover_test_bm25_top100.json
└── README.md

## Prerequisites

### 1. Install Python Dependencies
```bash
pip install elasticsearch tqdm
```

### 2. Install Docker Desktop

- **Download**: https://www.docker.com/products/docker-desktop/
- Install and launch Docker Desktop
- Wait for Docker to start (green indicator in menu bar/system tray)

### 3. Download HoVer Dataset

Download the HoVer dataset files and place them in the `data/` folder:
- `hover_train_release_v1.1.json`
- `hover_dev_release_v1.1.json`
- `hover_test_release_v1.1.json`

**Source**: [HoVer Dataset](https://github.com/hover-nlp/hover)

### 4. Download HotpotQA Wikipedia Dump

**Important**: This is a large download (~13 GB compressed, ~30 GB extracted)

1. Download from: https://hotpotqa.github.io/wiki-readme.html
   - File: `enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2`

2. Extract the archive:
```bash
   tar -xjf enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2
```

3. Place the extracted folder in `data/`:
```
   data/enwiki-20171001-pages-meta-current-withlinks-processed/
```

The folder should contain subdirectories `AA/`, `AB/`, ..., `FZ/`, each with `.bz2` files.

## Setup and Execution

### Step 1: Start Elasticsearch
```bash
docker run -d --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "ES_JAVA_OPTS=-Xms2g -Xmx2g" \
  elasticsearch:7.17.0
```

Wait ~30 seconds for Elasticsearch to start, then verify:
```bash
curl http://localhost:9200
```

You should see JSON output with Elasticsearch version info.

### Step 2: Index Wikipedia Documents

**One-time operation** (~30-60 minutes depending on hardware)
```bash
python3 run_indexing.py
```

This will:
- Create an Elasticsearch index named `hotpot_wiki`
- Index ~5.4 million Wikipedia documents
- Build BM25 inverted index for fast retrieval

**Progress**: You'll see updates every 10,000 documents indexed.

**Expected output**:
```
Starting Wikipedia indexing...
[Step 1/2] Creating Elasticsearch index...
Created index: hotpot_wiki

[Step 2/2] Indexing Wikipedia...
Indexed 10000 documents...
Indexed 20000 documents...
...
Successfully indexed 5486212 documents
✓ Indexing complete!
```

**Note**: The index persists in Elasticsearch, so you only need to do this once. If you restart Docker, use `docker start elasticsearch` to resume.

### Step 3: Run BM25 Retrieval

**Retrieves top-100 documents for each claim** (~20-30 minutes total)
```bash
python3 run_retrieval.py
```

This will:
- Process train, dev, and test sets
- Retrieve top-100 Wikipedia documents per claim using BM25
- Save results to `output/hover_{split}_bm25_top100.json`
- Evaluate retrieval quality (coverage and recall)

**Expected output**:
```
Processing TRAIN set
Loaded 18171 examples
Retrieving top-100 documents per claim...
✓ Saved results to: output/hover_train_bm25_top100.json

📊 Retrieval Metrics for train:
  Total claims: 18171
  Claims with ALL supporting docs: 5465
  Coverage: 30.08%
  Average Recall: 57.85%
```

## Results

### Retrieval Performance

| Dataset | Claims | Coverage | Avg Recall |
|---------|--------|----------|------------|
| **Train** | 18,171 | 30.08% | 57.85% |
| **Dev** | 4,000 | 19.80% | 51.75% |
| **Test** | 4,000 | N/A | N/A |

**Coverage**: Percentage of claims where ALL supporting documents were found in top-100

**Average Recall**: Average percentage of supporting documents found per claim

> **Note**: Lower coverage is expected for multi-hop reasoning tasks. BM25 relies on keyword matching without reasoning capabilities. These results are typical for single-stage retrieval on multi-hop datasets.

## Docker Commands Reference
```bash
# Check if Elasticsearch is running
docker ps

# View Elasticsearch logs
docker logs elasticsearch

# Stop Elasticsearch
docker stop elasticsearch

# Start Elasticsearch again (after stopping)
docker start elasticsearch

# Remove container (will delete index!)
docker stop elasticsearch
docker rm elasticsearch
```

## Implementation Details

### BM25 Parameters

- **k1**: 1.2 (term frequency saturation)
- **b**: 0.75 (length normalization)
- **Fields**: `title^3` (3x boost), `text^1`

### Why Multi-Hop is Challenging

HoVer requires finding multiple documents through reasoning chains:

**Example Claim**: "A hockey team calls Madison Square Garden its home..."

**Required Documents**:
1. "Madison Square Garden" → Rangers play there
2. "New York Rangers" → It's an NHL team
3. "NHL" → Info about other teams

**Challenge**: BM25 finds document 1 easily (direct keyword match), but documents 2-3 require reasoning that BM25 cannot perform.

### System Requirements

- **RAM**: 8GB minimum, 16GB recommended
- **Disk Space**: ~40GB free (for Wikipedia dump and index)
- **Time**: ~1-2 hours for complete setup and retrieval

## Output Format

Retrieval results are saved as JSON files with this structure:
```json
{
  "claim_uid": {
    "claim": "The claim text...",
    "retrieved_docs": [
      {
        "doc_id": "12345",
        "title": "Article Title",
        "sentences": ["Sentence 1", "Sentence 2", ...],
        "score": 42.5,
        "url": "https://en.wikipedia.org/wiki/Article_Title"
      }
    ],
    "label": "SUPPORTED",
    "supporting_facts": [["Article Title", 3], ...]
  }
}
```
