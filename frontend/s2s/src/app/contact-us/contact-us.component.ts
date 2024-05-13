import { Component } from '@angular/core';
import { FormGroup, FormControl, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { MessageService } from 'primeng/api';

@Component({
  selector: 'app-contact-us',
  templateUrl: './contact-us.component.html',
  styleUrl: './contact-us.component.css'
})
export class ContactUsComponent {
  showConfirmMessage: boolean = true;
  confirmMessage = "Thank you for contacting us! We will get back to you shortly."
  contactForm: FormGroup = new FormGroup({
    name: new FormControl('', [Validators.required]),
    email: new FormControl('', [Validators.required, Validators.email]),
    phoneNumber: new FormControl('', [Validators.required]),
    subject: new FormControl('', [Validators.required]),
    message: new FormControl('', [Validators.required]),
  });

  constructor(
    private router: Router,
    public messageService: MessageService
  ) {}

  /**
   * Resets the contact form.
   */
  resetForm() {
    this.contactForm.reset();
  }

  /**
   * Clears all messages from the message service.
   */
  clearMessages() {
    this.messageService.clear();
  }

  /**
   * Submits the contact form.
   */
  submitForm() {
    if (this.contactForm.valid) {
      this.clearMessages();
      this.messageService.add({severity: 'success', detail: this.confirmMessage});
      this.showConfirmMessage = true;
      this.resetForm();
    }
  }
}
