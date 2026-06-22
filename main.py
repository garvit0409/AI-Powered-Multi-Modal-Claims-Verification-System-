import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
import time
import pandas as pd
from agent import ClaimsAgent

def main():
    print("Initializing Claims Agent...")
    # Get API keys from environment
    gemini_key = os.environ.get("GEMINI_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")
    
    # Locate project directories
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Get reasoner provider and model from environment
    reasoner_provider = os.environ.get("REASONER_PROVIDER", "groq").lower()
    reasoner_model = os.environ.get("REASONER_MODEL")
    if not reasoner_model:
        reasoner_model = "gemma-4-31b-it" if reasoner_provider == "gemini" else "llama-3.1-8b-instant"

    # Instantiate agent
    agent = ClaimsAgent(
        gemini_key=gemini_key,
        groq_key=groq_key,
        repo_root=repo_root,
        reasoner_provider=reasoner_provider,
        reasoner_model=reasoner_model
    )
    
    input_csv = os.path.join(repo_root, "dataset", "claims.csv")
    output_csv_dataset = os.path.join(repo_root, "dataset", "output.csv")
    output_csv_root = os.path.join(repo_root, "output.csv")
    
    if not os.path.exists(input_csv):
        print(f"Error: Input file {input_csv} does not exist.")
        sys.exit(1)
        
    print(f"Reading claims from {input_csv}...")
    claims_df = pd.read_csv(input_csv)
    total_claims = len(claims_df)
    print(f"Loaded {total_claims} claims. Starting processing...")
    
    results = []
    for idx, row in claims_df.iterrows():
        print(f"[{idx+1}/{total_claims}] Processing claim for user {row['user_id']} ({row['claim_object']})...")
        try:
            res = agent.verify_claim(
                user_id=row['user_id'],
                image_paths=row['image_paths'],
                user_claim=row['user_claim'],
                claim_object=row['claim_object']
            )
            # Combine input and output columns
            res_row = {
                "user_id": row['user_id'],
                "image_paths": row['image_paths'],
                "user_claim": row['user_claim'],
                "claim_object": row['claim_object'],
                "evidence_standard_met": str(res.get("evidence_standard_met", False)).lower(),
                "evidence_standard_met_reason": res.get("evidence_standard_met_reason", "none"),
                "risk_flags": res.get("risk_flags", "none"),
                "issue_type": res.get("issue_type", "unknown"),
                "object_part": res.get("object_part", "unknown"),
                "claim_status": res.get("claim_status", "not_enough_information"),
                "claim_status_justification": res.get("claim_status_justification", "none"),
                "supporting_image_ids": res.get("supporting_image_ids", "none"),
                "valid_image": str(res.get("valid_image", True)).lower(),
                "severity": res.get("severity", "unknown")
            }
            results.append(res_row)
        except Exception as e:
            print(f"Error processing claim {idx+1}: {e}")
            res_row = {
                "user_id": row['user_id'],
                "image_paths": row['image_paths'],
                "user_claim": row['user_claim'],
                "claim_object": row['claim_object'],
                "evidence_standard_met": "false",
                "evidence_standard_met_reason": f"Processing error: {str(e)}",
                "risk_flags": "manual_review_required",
                "issue_type": "unknown",
                "object_part": "unknown",
                "claim_status": "not_enough_information",
                "claim_status_justification": f"Processing error: {str(e)}",
                "supporting_image_ids": "none",
                "valid_image": "false",
                "severity": "unknown"
            }
            results.append(res_row)

            
    results_df = pd.DataFrame(results)
    
    # Save to both locations
    results_df.to_csv(output_csv_dataset, index=False)
    results_df.to_csv(output_csv_root, index=False)
    print(f"Successfully processed all claims. Predictions written to {output_csv_root} and {output_csv_dataset}.")

if __name__ == "__main__":
    main()
