import os
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from segment_depth.data import SAM2Dataset
from transformers import SamModel, SamProcessor, AutoImageProcessor
from transformers import AutoModelForImageSegmentation

# Distillation model using Lightning
class SAM2Distiller(pl.LightningModule):
    def __init__(self, teacher_model_id="facebook/sam2", student_model_id="facebook/sam2-small"):
        super().__init__()
        self.teacher = SamModel.from_pretrained(teacher_model_id)
        self.student = AutoModelForImageSegmentation.from_pretrained(student_model_id)
        self.processor = SamProcessor.from_pretrained(teacher_model_id)
        
        # Freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def forward(self, pixel_values, input_points):
        return self.student(pixel_values=pixel_values, input_points=input_points)
    
    def training_step(self, batch, batch_idx):
        # Process with teacher model
        with torch.no_grad():
            teacher_outputs = self.teacher(**batch["inputs"])
        
        # Process with student model
        student_outputs = self.student(**batch["inputs"])
        
        # Compute distillation loss
        loss = torch.nn.functional.mse_loss(
            student_outputs.pred_masks, 
            teacher_outputs.pred_masks
        )
        
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.student.parameters(), lr=1e-4)