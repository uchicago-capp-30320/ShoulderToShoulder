import { Component } from '@angular/core';
import { FormGroup, FormControl, Validators } from '@angular/forms';
import { catchError } from 'rxjs/operators';
import { EMPTY } from 'rxjs';

// services
import { AuthService } from '../_services/auth.service';
import { OnboardingService } from '../_services/onboarding.service';

// helpers
import { StrongPasswordRegx } from '../_helpers/patterns';
import { confirmPasswordValidator, differentPasswordValidator } from '../_helpers/validators';
import { PasswordChange } from '../_models/password-change';

@Component({
  selector: 'app-profile-settings',
  templateUrl: './profile-settings.component.html',
  styleUrl: './profile-settings.component.css'
})
export class ProfileSettingsComponent {
  showOnboardingDialog = false;
  user = this.authService.userValue;

  changePasswordForm = new FormGroup({
    username: new FormControl(this.user.username, Validators.required),
    currentPassword: new FormControl('', Validators.required),
    password: new FormControl('', [
      Validators.required, 
      Validators.minLength(8), 
      Validators.pattern(StrongPasswordRegx),
      differentPasswordValidator]),
    confirmPassword: new FormControl('', [Validators.required, confirmPasswordValidator])
  },
  {
    validators: [differentPasswordValidator, confirmPasswordValidator]
  });

  updateUserInformationForm = new FormGroup({
    firstName: new FormControl(this.user.first_name, Validators.required),
    lastName: new FormControl(this.user.last_name, Validators.required),
    email: new FormControl(this.user.email, Validators.required),
  });

  constructor(
    private authService: AuthService,
    private onboardingService: OnboardingService
  ) {}

  openOnboardingDialog() {
    this.showOnboardingDialog = true;
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

  submitNewPassword() {
    console.log('submitting new password')
    let username = this.changePasswordForm.get('username')?.value;
    let currentPassword = this.changePasswordForm.get('currentPassword')?.value;
    let password = this.changePasswordForm.get('password')?.value;
    let confirmPassword = this.changePasswordForm.get('confirmPassword')?.value;

    let passwordChange: PasswordChange = {
      username: username ? username : '',
      current_password: currentPassword ? currentPassword : '',
      password: password ? password : '',
      confirm_password: confirmPassword ? confirmPassword : ''
    };

    this.authService.changePassword(passwordChange).pipe(
      catchError(error => {
        console.error('Error changing password:', error);
        return EMPTY;
      })
    ).subscribe(() => {
      console.log('Password changed successfully');
      this.changePasswordForm.reset();
    });
    this.changePasswordForm.reset();
  }

  submitOnboardingChanges() {
    console.log('submitting onboarding changes');
    this.showOnboardingDialog = false;
    this.onboardingService.updateOnboarding();
  }
}
