import { Component } from '@angular/core';
import { FormGroup, FormControl, Validators, ReactiveFormsModule } from '@angular/forms';

// helpers
import { StrongPasswordRegx } from '../_helpers/patterns';

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
  signupForm: FormGroup = new FormGroup({
    firstName: new FormControl('', [Validators.required]),
    lastName: new FormControl('', [Validators.required]),
    phoneNum: new FormControl('', [Validators.required, Validators.pattern('^[0-9]*$'), Validators.minLength(10), Validators.maxLength(10)]),
    email: new FormControl('', [Validators.required, Validators.email]),
    password: new FormControl('', [Validators.required, Validators.minLength(8), Validators.pattern(StrongPasswordRegx)]),
    confirmPassword: new FormControl('')
  });

  /**
   * Getter for the first password form field. Used for strong password 
   * validation.
   */
  get passwordFormField() {
    return this.signupForm.get('password');
  }

  /**
   * Handles form submission.
   */
  onSubmit() {
    console.log(this.signupForm.value);
  }

  addUser() {
    console.log(this.signupForm.value);
  }

}
