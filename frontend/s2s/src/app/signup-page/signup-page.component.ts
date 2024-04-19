import { Component} from '@angular/core';
import { FormGroup, FormControl, Validators } from '@angular/forms';
import { Router } from '@angular/router';

// helpers
import { StrongPasswordRegx } from '../_helpers/patterns';
import { confirmPasswordValidator } from '../_helpers/validators';

/**
 * Implements the application's Sign Up page, including the sign-up form.
 * 
 * The sign-up form includes the following fields:
 * - First Name (required)
 * - Last Name (required)
 * - Phone Number (optional, must be 10 digits)
 * - Email (required, must be a valid email address)
 * - Password (required, must be at least 8 characters long and contain at least 
 *             one uppercase letter, one lowercase letter, one number, and one 
 *             special character)
 * - Confirm Password (required, must match the Password field)
 * 
 * The form includes the following functionality:
 * - Validation for all fields
 * - Custom validation for the Password and Confirm Password fields
 * - Toggling the visibility of the password field
 * - Resetting the form
 * - Handling form submission
 * 
 * @author Chanteria Milner
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
    phoneNumber: new FormControl('', [
                                      Validators.pattern('^[0-9]*$'), 
                                      Validators.minLength(10), 
                                      Validators.maxLength(10)]),
    email: new FormControl('', [Validators.required, Validators.email]),
    password: new FormControl('', [
                                      Validators.required, 
                                      Validators.minLength(8), 
                                      Validators.pattern(StrongPasswordRegx)]),
    confirmPassword: new FormControl('', [
                                          Validators.required, 
                                          confirmPasswordValidator]),
  },
  {
    validators: confirmPasswordValidator
  });

  constructor(
    private route: Router
  ) {}

  /**
   * Resets the sign up form.
   */
  resetForm() {
    this.signupForm.reset();
  }

  /**
   * Returns the form control for the specified field.
   * 
   * @param fieldName The name of the field.
   */
  getFormControl(fieldName: string) {
    return this.signupForm.get(fieldName);
  }

  /**
   * Toggles the visibility of the password field.
   * 
   * @param inputField The password field.
   */
  togglePasswordField(inputField: HTMLInputElement): void {
    const type = inputField.type;
    inputField.type = type === 'password' ? 'text' : 'password';
  }

  /**
   * Handles form submission.
   * TODO: Implement form submission logic once the backend can handle user
   *       registration.
   */
  onSubmit() {
    // check if the form is valid
    if (this.signupForm.invalid) {

      // if the form is invalid, mark all fields as touched
      this.signupForm.markAllAsTouched();
      return;
    }
    this.resetForm();

    this.route.navigate(['/onboarding']);
  }
}
