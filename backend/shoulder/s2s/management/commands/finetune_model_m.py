"""
Custom migration command for fine-tuning the ML model.
"""

from django.core.management.base import BaseCommand
from s2s.views import SuggestionResultsViewSet
import datetime

class Command(BaseCommand):
    help = "Fine-tunes the ML model."

    def handle(self, *args, **kwargs):
        viewset = SuggestionResultsViewSet()
        result = viewset.perform_finetune()
        if "error" in result:
            self.stderr.write(self.style.ERROR(f"Error: {result['error']}"))
        else:
            self.stdout.write(self.style.SUCCESS(f"Finetuning completed successfully. Results:"))
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result_message = \
            f"""Time: {current_time}

            Pre-training:
            Number of pre-training epochs: {result['pretraining_epochs'][-1]}
            Pre-training loss: from {result['pretraining_loss_list'][0]} to {result['pretraining_loss_list'][-1]}
            Pre-training accuracy: from {result['pretraining_acc_list'][0]} to {result['pretraining_acc_list'][-1]}

            Fine-tuning:
            Number of finetuning epochs: {result['finetuning_epochs'][-1]}
            Finetuning loss: from {result['finetuning_loss_list'][0]} to {result['finetuning_loss_list'][-1]}
            Finetuning accuracy: from {result['finetuning_acc_list'][0]} to {result['finetuning_acc_list'][-1]}
            """
            self.stdout.write(self.style.SUCCESS(result_message))