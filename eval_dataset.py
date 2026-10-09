from typing import TypedDict, List

class EvalSample(TypedDict):
    question: str
    ground_truth: str
    category: str # direct_fact, paraphrase, adversarial

GOLDEN_EVAL_DATASET: List[EvalSample] = [
    # 1. Direct factual questions (testing retrieval precision)
  {
    "category": "direct_fact",
    "question": "What is the required encryption standard for customer data at rest?",
    "ground_truth": "Customer data must be encrypted using AES-256 at rest."
  },
  {
    "category": "direct_fact",
    "question": "What are the specific authentication steps required to access production databases?",
    "ground_truth": "Accessing production databases requires connecting via the corporate VPN followed by a hardware security key (YubiKey) 2FA verification."
  },

    # 2. Semantic paraphrases (testing embedded generalization)
  {
    "category": "paraphrase",
    "question": "If I only spend $1200 on my laptop this year, how much of that leftover money can I add to my allowance for next year?",
    "ground_truth": "A maximum of $500 of unused hardware budget can roll over to the following year."
  },
  {
    "category": "paraphrase",
    "question": "Even though we are a remote team, are there any mandatory periods during the day when I have to be at my computer?",
    "ground_truth": "Yes, all engineers must be online during the core overlap hours of 10:00 AM to 3:00 PM EST."
  },

    # 3. Out of domain (testing hallucination defenses)
  {
    "category": "adversarial",
    "question": "What is the company's matching policy for 401k retirement plans?",
    "ground_truth": "I cannot answer this based on the provided documents."
  },
  {
    "category": "adversarial",
    "question": "How many days of Paid Time Off (PTO) or sick leave do junior engineers get per year?",
    "ground_truth": "I cannot answer this based on the provided documents."
  }
]