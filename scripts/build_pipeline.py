# scripts/build_pipeline.py

from src.model import train_model
from src.explain import run_explainability
from src.simulation import run_simulation
from src.rag import run_rag

def run_pipeline():
    print("Step 1: Training model...")
    train_model()

    print("Step 2: Explainability...")
    run_explainability()

    print("Step 3: Simulation...")
    run_simulation()

    print("Step 4: RAG...")
    run_rag()

    print("Pipeline complete.")

if __name__ == "__main__":
    run_pipeline()