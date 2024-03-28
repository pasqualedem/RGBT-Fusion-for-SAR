from transformers import Trainer, TrainingArguments

from sarfusion.data import get_trainval
from sarfusion.models import build_model
from sarfusion.models.loss import build_loss
from sarfusion.models.wrapper import ModelWrapper


class Run:
    def __init__(self):
        pass
    
    def init(self, params):
        self.params = params
        self.model = build_model(self.params['model'])
        self.loss = build_loss(self.params.get('loss', {}))
        self.model = ModelWrapper(self.model, self.loss)
        
    def _prep_for_train(self):
        self.data = get_trainval(self.params['dataset'], self.params['dataloader'])
        self.train_params = self.params['train']
    
    def launch(self):
        return self.train()

    def train(self):
        arguments = TrainingArguments(
            report_to="wandb",
            output_dir=self.params['tracker']['output_dir'],
            per_device_train_batch_size=self.params['dataloader']['batch_size'],
            dataloader_num_workers=self.params['dataloader']['num_workers'],
        )
        trainset, valset = self.data
        collate_fn = trainset.collate_fn if hasattr(trainset, 'collate_fn') else None
        trainer = Trainer(
            model=self.model,
            args=arguments,
            train_dataset=trainset,
            data_collator=collate_fn,
            eval_dataset=valset,
        )
        trainer.train()
        return trainer