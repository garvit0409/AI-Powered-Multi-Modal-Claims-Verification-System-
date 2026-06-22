# Operational and Evaluation Report

Generated on: 2026-06-19
Dataset: `dataset/sample_claims.csv`
Number of Claims Processed: 20
Number of Images Processed: 29

---

## 1. System Evaluation Metrics

We evaluated the performance of our multi-modal claims verification pipeline against the ground truth labels in `dataset/sample_claims.csv`.

| Metric / Attribute | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 Score |
|---|---|---|---|---|
| **Claim Status** | 70.00% | 66.11% | 67.22% | 64.31% |
| **Evidence Standard Met** | 70.00% | 60.44% | 68.63% | 60.00% |
| **Issue Type** | 75.00% | 76.83% | 79.17% | 74.74% |
| **Object Part** | 90.00% | 85.29% | 85.29% | 84.31% |
| **Valid Image** | 95.00% | 83.33% | 97.22% | 88.57% |
| **Severity** | 75.00% | 59.11% | 67.88% | 62.00% |

---

## 2. Operational & Cost Analysis

### Model Token Usage

| Provider / Model | Input Tokens | Output Tokens | Cost (USD) |
|---|---|---|---|
| **Gemini (gemma-4-31b-it)** | 0 | 0 | $0.000000 |
| **Groq (meta-llama/llama-4-scout-17b-16e-instruct)** | 82,101 | 9,543 | $0.018329 |
| **Total** | | | **$0.018329** |

### Pricing Assumptions
* **Gemini (gemma-4-31b-it)**: Input: $0.075 / 1M tokens, Output: $0.30 / 1M tokens
* **Groq (meta-llama/llama-4-scout-17b-16e-instruct)**: Input: $0.20 / 1M tokens, Output: $0.20 / 1M tokens

### Performance & Latency
* **Total Runtime**: 405.55 seconds
* **Average Latency per Claim**: 20.28 seconds
* **Images Processed**: 29

---


## 3. Production Scaling Considerations (TPM / RPM)

For a production deployment running over large datasets, the following rate limits apply:
1. **Groq API Rate Limits**:
   - Typically 100 requests per minute (RPM) and 100,000 tokens per minute (TPM).
   - Our pipeline processes images sequentially. For claims with multiple images, we recommend a throttling delay of `0.5s` between calls or using an asynchronous processing queue to prevent `429 Rate Limit` errors.
2. **Gemini API Rate Limits**:
   - The Gemini 2.5 Flash model has a high capacity (typically 15 RPM for free tier, up to 2000 RPM for pay-as-you-go).
3. **Mitigation Strategy**:
   - **Retries**: Implement exponential backoff when a `429` status code is encountered.
   - **Caching**: Cache image feature extraction results (e.g. hash of the image file) so that if the same image is re-submitted or evaluated twice, the VLM call is bypassed completely.
