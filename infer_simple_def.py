import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import lightning.pytorch as pl

device = "cpu"
class args:
    inference_checkpoint="./model/lightning_model_2input.ckpt"
    checkpoint_path="./model/mint.ckpt"
    csv_file="./data/egfr_2input.csv"
    cfg="./model/esm2_t33_650M_UR50D.json"
    sep_chains=False

from utils_2input import *

"""Inference mode: Load model checkpoint and run predictions."""
torch.manual_seed(CFG.SEED)
np.random.seed(CFG.SEED)
random.seed(CFG.SEED)

# Load configuration from JSON
CONFIG_DICT_PATH = args.cfg
cfg = argparse.Namespace()
with open(CONFIG_DICT_PATH) as f:
    cfg.__dict__.update(json.load(f))

embedder = SequenceEmbedder(
    cfg,
    args.checkpoint_path,
    freeze_percent=1.0,
    use_multimer=True,
    sep_chains=args.sep_chains,
    device=device
)

# Initialize MLP head
mlp_head = MLP(
    input_dim=CFG.EMBED_DIM,
    output_dim=CFG.OUTPUT_DIM,
    lr=CFG.LR,
    weight_decay=CFG.WD
)

# Initialize combined model
model = CombinedModel(embedder, mlp_head)
model = model.to(device)
loss_fn = torch.nn.HuberLoss()

# Load the trained LightningModule checkpoint
lightning_model = DeltaGLightningModule.load_from_checkpoint(
    checkpoint_path=args.inference_checkpoint,
    model=model,
    loss_fn=loss_fn,
    lr=CFG.LR,
    weight_decay=CFG.WD,
    gamma=0.95,
    strict=False
)

lightning_model.eval()

# Create a Trainer for inference
trainer = pl.Trainer(
    accelerator='auto',
    # strategy=DDPStrategy(find_unused_parameters=True),
    logger=False
)


# Expect the CSV for inference (should contain sequences and optionally targets)
def predict_fn(seq1, seq_mut):
    # seq1 = ["MARTKQTARKSA"]
    # seq_mut = ["MARTKQTARKSA"]

    df = pd.DataFrame({
        "col1": seq1,
        "col2": seq_mut,
        "target_col": [0]  # Dummy target for inference
    })

    dataset = ProteinDFDataset(df, "col1", "col2", "target_col")

    dataloader = DataLoader(
        dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        collate_fn=MutationalPPICollateFn(),
        num_workers=7
    )



        # Run inference using trainer.predict
    predictions = trainer.predict(lightning_model, dataloaders=dataloader)


    results = []
    for batch in predictions:
        for pred in batch:
            val = pred.detach().cpu().item()
            results.append(val)
    return results

res = predict_fn("MARTKQTARKSA", "MAAAKQTARKSA")
print(res)
