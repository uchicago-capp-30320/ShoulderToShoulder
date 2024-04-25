import { Component} from '@angular/core';
import { FormGroup, FormControl, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { catchError } from 'rxjs/operators';
import { EMPTY } from 'rxjs';

// services
import { AuthService } from '../_services/auth.service';

// models
import { UserLogIn } from '../_models/user';

// helpers

/**
 * Implements the application's Log In page.
 */
@Component({
  selector: 'app-login-page',
  templateUrl: './log-in.component.html',
  styleUrl: './log-in.component.css'
})
export class LogInComponent {
  /**
   * The sign-up form.
   */
  // Initialize the form group
  loginForm: FormGroup = new FormGroup({
    email: new FormControl('', [Validators.required, Validators.email]),
    password: new FormControl('', [
                                      Validators.required,]),
  });

  constructor(
    private route: Router,
    private authService: AuthService
  ) {}

  /**
   * Resets the sign up form.
   */
  resetForm() {
    this.loginForm.reset();
  }

  /**
   * Returns the form control for the specified field.
   * 
   * @param fieldName The name of the field.
   */
  getFormControl(fieldName: string) {
    return this.loginForm.get(fieldName);
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
   * Handles submission for the log in form.
   */
  onSubmit() {
    // check if the form is valid
    if (this.loginForm.invalid) {

      // if the form is invalid, mark all fields as touched
      this.loginForm.markAllAsTouched();
      return;
    }

    // use auth service to sign user up
    const user: UserLogIn = {
      username: this.getFormControl('email')?.value,
      password: this.getFormControl('password')?.value,
    };

    // log in user
    this.authService.login(user).pipe(
      catchError((error) => {
        console.error('Error logging in user:', error);
        return EMPTY;
      }),
    ).subscribe(() => {
      this.resetForm();
      this.route.navigate(['/onboarding']);
    });
  }
}
