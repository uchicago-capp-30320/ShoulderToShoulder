import { Component} from '@angular/core';
import { FormGroup, FormControl, Validators } from '@angular/forms';

// helpers
import { StrongPasswordRegx } from '../_helpers/patterns';
import { confirmPasswordValidator } from '../_helpers/validators';

/**
 * Implements the application's Sign Up page, including the sign-up form.
 */
@Component({
  selector: 'app-signup-page',
  templateUrl: './signup-page.component.html',
  styleUrl: './signup-page.component.css'
})
export class SignupPageComponent {
  /**
   * The sign-up form.
   */
  // Initialize the form group
  signupForm: FormGroup = new FormGroup({
    firstName: new FormControl('', [Validators.required]),
    lastName: new FormControl('', [Validators.required]),
    phoneNumber: new FormControl('', [Validators.pattern('^[0-9]*$'), Validators.minLength(10), Validators.maxLength(10)]),
    email: new FormControl('', [Validators.required, Validators.email]),
    password: new FormControl('', [Validators.required, Validators.minLength(8), Validators.pattern(StrongPasswordRegx)]),
    confirmPassword: new FormControl('', [Validators.required, confirmPasswordValidator]),
  },
  {
    validators: confirmPasswordValidator
  });

  /**
   * Resets the sign up form.
   */
  resetForm() {
    this.signupForm.reset();
  }

  /**
   * Handles form submission.
   */
  onSubmit() {
    // Check if the form is valid
    if (this.signupForm.invalid) {
      // If the form is invalid, mark all fields as touched
      this.signupForm.markAllAsTouched();
      return;
    }
    console.log(this.signupForm.value);
    this.resetForm();
  }

}
