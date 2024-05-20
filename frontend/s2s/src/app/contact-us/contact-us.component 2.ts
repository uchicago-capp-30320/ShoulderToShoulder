import { Component } from '@angular/core';
import { FormGroup, FormControl, Validators } from '@angular/forms';
import { Router } from '@angular/router';

@Component({
  selector: 'app-contact-us',
  templateUrl: './contact-us.component.html',
  styleUrl: './contact-us.component.css'
})
export class ContactUsComponent {
  contactForm: FormGroup = new FormGroup({
    name: new FormControl('', [Validators.required]),
    email: new FormControl('', [Validators.required, Validators.email]),
    phoneNumber: new FormControl('', [Validators.required]),
    subject: new FormControl('', [Validators.required]),
    message: new FormControl('', [Validators.required]),
  });

  constructor(
    private router: Router,
  ) {}

  /**
   * Resets the contact form.
   */
  resetForm() {
    this.contactForm.reset();
  }

  /**
   * Submits the contact form.
   */
  submitForm() {
    if (this.contactForm.valid) {
      let mailto_link = 'mailto:shouldertoshoulder.contact@gmail.com?subject=' +
      "[Contact Form] " + encodeURIComponent(this.contactForm.value.subject) +
      '&body=' +
      encodeURIComponent(this.contactForm.value.message) +
      '%0A%0A' +
      encodeURIComponent(this.contactForm.value.name) +
      '%0A' +
      this.formatPhoneNumber(encodeURIComponent(this.contactForm.value.phoneNumber)) +
      '%0A' +
      encodeURIComponent(this.contactForm.value.email);
      window.open(mailto_link, '_blank');

      this.resetForm();
    }
  }

  /**
   * Formats the phone number.
   */
  formatPhoneNumber(phoneNumber: string) {
    let formattedPhoneNumber = phoneNumber.replace(/(\d{3})(\d{3})(\d{4})/, '($1) $2-$3');
    return formattedPhoneNumber;
  }
}
