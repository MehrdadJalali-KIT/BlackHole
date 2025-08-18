import json
import logging

logging.basicConfig(
    level=logging.INFO,
    filename="bh_evaluation.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def reset_checkpoint():
    checkpoint_file = "bh_evaluation_checkpoint.json"
    checkpoint = {
        "completed_sparsification": [],
        "completed_evaluation": [],
        "results": [],
        "fixed_test_nodes": []
    }
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)
        logger.info("Checkpoint reset successfully")
    except Exception as e:
        logger.error(f"Failed to reset checkpoint: {e}")

if __name__ == "__main__":
    reset_checkpoint()