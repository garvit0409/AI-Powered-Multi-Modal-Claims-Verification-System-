import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
import time
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add parent directory to path so agent can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agent import ClaimsAgent

def evaluate():
    print("Initializing Claims Agent for Evaluation...")
    # Get API keys from environment
    gemini_key = os.environ.get("GEMINI_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")
    
    # Locate project directories
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    # Get reasoner provider and model from environment
    reasoner_provider = os.environ.get("REASONER_PROVIDER", "groq").lower()
    reasoner_model = os.environ.get("REASONER_MODEL")
    if not reasoner_model:
        reasoner_model = "gemini-2.5-flash-lite" if reasoner_provider == "gemini" else "llama-3.1-8b-instant"

    agent = ClaimsAgent(
        gemini_key=gemini_key,
        groq_key=groq_key,
        repo_root=repo_root,
        reasoner_provider=reasoner_provider,
        reasoner_model=reasoner_model
    )
    
    sample_csv = os.path.join(repo_root, "dataset", "sample_claims.csv")
    if not os.path.exists(sample_csv):
        print(f"Error: Sample file {sample_csv} not found.")
        sys.exit(1)
        
    print(f"Reading sample claims from {sample_csv}...")
    sample_df = pd.read_csv(sample_csv)
    
    # Clear prior token counts
    agent.reset_token_usage()
    
    predictions = []
    start_time = time.time()
    
    total_claims = len(sample_df)
    for idx, row in sample_df.iterrows():
        print(f"Evaluating [{idx+1}/{total_claims}] for user {row['user_id']} ({row['claim_object']})...")
        try:
            res = agent.verify_claim(
                user_id=row['user_id'],
                image_paths=row['image_paths'],
                user_claim=row['user_claim'],
                claim_object=row['claim_object']
            )
            predictions.append(res)
        except Exception as e:
            print(f"Error evaluating row {idx+1}: {e}")
            predictions.append({
                "evidence_standard_met": False,
                "risk_flags": "manual_review_required",
                "issue_type": "unknown",
                "object_part": "unknown",
                "claim_status": "not_enough_information",
                "valid_image": False,
                "severity": "unknown"
            })

            
    total_latency = time.time() - start_time
    
    pred_df = pd.DataFrame(predictions)
    
    # Clean boolean columns for accurate comparison
    def clean_bool(val):
        if isinstance(val, bool):
            return val
        return str(val).lower() == 'true'
        
    y_true_met = sample_df['evidence_standard_met'].apply(clean_bool)
    y_pred_met = pred_df['evidence_standard_met'].apply(clean_bool)
    
    y_true_status = sample_df['claim_status'].str.lower()
    y_pred_status = pred_df['claim_status'].str.lower()
    
    y_true_issue = sample_df['issue_type'].str.lower()
    y_pred_issue = pred_df['issue_type'].str.lower()
    
    y_true_part = sample_df['object_part'].str.lower()
    y_pred_part = pred_df['object_part'].str.lower()
    
    y_true_valid = sample_df['valid_image'].apply(clean_bool)
    y_pred_valid = pred_df['valid_image'].apply(clean_bool)
    
    y_true_sev = sample_df['severity'].str.lower()
    y_pred_sev = pred_df['severity'].str.lower()
    
    # Calculate metrics
    acc_met = accuracy_score(y_true_met, y_pred_met)
    acc_status = accuracy_score(y_true_status, y_pred_status)
    acc_issue = accuracy_score(y_true_issue, y_pred_issue)
    acc_part = accuracy_score(y_true_part, y_pred_part)
    acc_valid = accuracy_score(y_true_valid, y_pred_valid)
    acc_sev = accuracy_score(y_true_sev, y_pred_sev)
    
    # Precision, Recall, F1 for all attributes
    def get_p_r_f1(y_true, y_pred):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        return p, r, f

    prec_status, rec_status, f1_status = get_p_r_f1(y_true_status, y_pred_status)
    prec_met, rec_met, f1_met = get_p_r_f1(y_true_met, y_pred_met)
    prec_issue, rec_issue, f1_issue = get_p_r_f1(y_true_issue, y_pred_issue)
    prec_part, rec_part, f1_part = get_p_r_f1(y_true_part, y_pred_part)
    prec_valid, rec_valid, f1_valid = get_p_r_f1(y_true_valid, y_pred_valid)
    prec_sev, rec_sev, f1_sev = get_p_r_f1(y_true_sev, y_pred_sev)
    
    # Pricing assumptions:
    # Gemini 2.5 Flash: Input $0.075 / 1M tokens ($0.000000075 / token), Output $0.30 / 1M tokens ($0.0000003 / token)
    # Groq (Llama Vision): Input $0.20 / 1M tokens ($0.0000002 / token), Output $0.20 / 1M tokens ($0.0000002 / token)
    gemini_in_cost = agent.token_usage["gemini_input"] * 0.075 / 1_000_000
    gemini_out_cost = agent.token_usage["gemini_output"] * 0.30 / 1_000_000
    groq_in_cost = agent.token_usage["groq_input"] * 0.20 / 1_000_000
    groq_out_cost = agent.token_usage["groq_output"] * 0.20 / 1_000_000
    total_cost = gemini_in_cost + gemini_out_cost + groq_in_cost + groq_out_cost
    
    # Generate the Markdown report
    report_md = f"""# Operational and Evaluation Report

Generated on: 2026-06-19
Dataset: `dataset/sample_claims.csv`
Number of Claims Processed: {total_claims}
Number of Images Processed: {agent.token_usage["images_processed"]}

---

## 1. System Evaluation Metrics

We evaluated the performance of our multi-modal claims verification pipeline against the ground truth labels in `dataset/sample_claims.csv`.

| Metric / Attribute | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 Score |
|---|---|---|---|---|
| **Claim Status** | {acc_status:.2%} | {prec_status:.2%} | {rec_status:.2%} | {f1_status:.2%} |
| **Evidence Standard Met** | {acc_met:.2%} | {prec_met:.2%} | {rec_met:.2%} | {f1_met:.2%} |
| **Issue Type** | {acc_issue:.2%} | {prec_issue:.2%} | {rec_issue:.2%} | {f1_issue:.2%} |
| **Object Part** | {acc_part:.2%} | {prec_part:.2%} | {rec_part:.2%} | {f1_part:.2%} |
| **Valid Image** | {acc_valid:.2%} | {prec_valid:.2%} | {rec_valid:.2%} | {f1_valid:.2%} |
| **Severity** | {acc_sev:.2%} | {prec_sev:.2%} | {rec_sev:.2%} | {f1_sev:.2%} |

---

## 2. Operational & Cost Analysis

### Model Token Usage

| Provider / Model | Input Tokens | Output Tokens | Cost (USD) |
|---|---|---|---|
| **Gemini ({agent.gemini_model})** | {agent.token_usage["gemini_input"]:,} | {agent.token_usage["gemini_output"]:,} | ${gemini_in_cost + gemini_out_cost:.6f} |
| **Groq ({agent.groq_model})** | {agent.token_usage["groq_input"]:,} | {agent.token_usage["groq_output"]:,} | ${groq_in_cost + groq_out_cost:.6f} |
| **Total** | | | **${total_cost:.6f}** |

### Pricing Assumptions
* **Gemini ({agent.gemini_model})**: Input: $0.075 / 1M tokens, Output: $0.30 / 1M tokens
* **Groq ({agent.groq_model})**: Input: $0.20 / 1M tokens, Output: $0.20 / 1M tokens

### Performance & Latency
* **Total Runtime**: {total_latency:.2f} seconds
* **Average Latency per Claim**: {total_latency / total_claims:.2f} seconds
* **Images Processed**: {agent.token_usage["images_processed"]}

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
"""
    
    report_path = os.path.join(repo_root, "code", "evaluation", "evaluation_report.md")
    with open(report_path, "w") as f:
        f.write(report_md)
        
    print(f"Evaluation report written successfully to {report_path}")

if __name__ == "__main__":
    evaluate()
