
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import SamProcessor
from segment_depth.source import SAM2Dataset, SAM2Distiller
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Main training function
def train_distilled_sam2(data_dir, output_dir, batch_size=8, max_epochs=10):
    # Initialize model and processor
    model_id = "facebook/sam2"  # Use appropriate SAM2 model ID
    processor = SamProcessor.from_pretrained(model_id)
    
    # Create dataset and dataloader
    dataset = SAM2Dataset(data_dir, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Initialize model
    model = SAM2Distiller(teacher_model_id=model_id)
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="sam2-distilled-{epoch:02d}-{train_loss:.2f}",
        monitor="train_loss",
        save_top_k=3,
        mode="min"
    )
    
    early_stop_callback = EarlyStopping(
        monitor="train_loss",
        patience=3,
        mode="min"
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",  # Use GPU if available
        callbacks=[checkpoint_callback, early_stop_callback]
    )
    
    # Train the model
    trainer.fit(model, dataloader)
    
    # Save the final student model
    model.student.save_pretrained(os.path.join(output_dir, "final_model"))
    processor.save_pretrained(os.path.join(output_dir, "processor"))
    
    print(f"Distilled model saved to {output_dir}")



if __name__ == "__main__":
    train_distilled_sam2(
        data_dir="path/to/your/images",
        output_dir="path/to/save/model",
        batch_size=8,
        max_epochs=10
    )