# Automated Software Requirement Classification

The goal of this project to classify requirements using gemini with a hierarchical structure 
At first : Requirement are classified as main types (Functional, Quality, Constraint, Business)
Second: Requirements are subdivided into its types 

Three approaches are used

1. **BERT** - fine-tuning `bert-base-uncased` on the labelled datasets
2. **LLM prompting** - Gemini with zero-shot, few-shot, chain-of-thought 
3. **Hybrid RAG framework (ReqClassify)** - Gemini and labelled examples and ISO/IEC 25010 and
   ISO/IEC/IEEE 29148 definitions stored in Supabase (pgvector), with a Streamlit app


## Folder structure

```
1_data/            datasets (labelled and unlabelled csv files)
2_experiments/
    bert/scripts/  BERT training scripts 
    llm/gemini/    prompting scripts, prompts and evaluation
4_utils/           helper scripts
5_Automation/      ReqClassify app (streamlit), RAG layer and ISO upload scripts
```

## Setup

Python 3.10+ is needed.

```

## Notes

- Trained models and result files are not in this repository because of the size.
- Results in the thesis were made with `temperature = 0` and `seed = 42`.
Flow of the  Retrieval- Augmented Generation representation is given like this :
 
