import { Component} from '@angular/core';
import { FormGroup, FormControl, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { catchError } from 'rxjs/operators';
import { EMPTY } from 'rxjs';

// services
import { AuthService } from '../_services/auth.service';

// models
import { UserSignUp } from '../_models/user';

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

  isLoading: boolean = false;
  showError = false;
  errorMessage = '';

  constructor(
    private route: Router,
    private authService: AuthService
  ) {
  }

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
   * Handles submission for the sign up form.
   */
  onSubmit() {
    // check if the form is valid
    if (this.signupForm.invalid) {

      // if the form is invalid, mark all fields as touched
      this.signupForm.markAllAsTouched();
      return;
    }

    // use auth service to sign user up
    const user: UserSignUp = {
      first_name: this.getFormControl('firstName')?.value,
      last_name: this.getFormControl('lastName')?.value,
      email: this.getFormControl('email')?.value,
      password: this.getFormControl('password')?.value,
    };

    // sign up user
    this.isLoading = true;
    this.authService.signup(user).pipe(
      catchError((error) => {
        console.error('Error signing up user:', error);
        if (error.status === 400) {
          this.errorMessage = 'Email already exists. Please log in.';
          this.showError = true;
        }
        this.isLoading = false;
        return EMPTY;
      }),
    ).subscribe(() => {
      this.resetForm();
      this.isLoading = false;
      this.route.navigate(['/onboarding']);
    });
  }
}