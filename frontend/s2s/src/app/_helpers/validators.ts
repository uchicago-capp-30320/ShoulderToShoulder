import { AbstractControl, ValidationErrors, ValidatorFn } from '@angular/forms';
import { ConfirmEventType } from 'primeng/api';

export const confirmPasswordValidator: ValidatorFn = (control: AbstractControl): ValidationErrors | null => {
  const password = control.get('password');
  const confirmPassword = control.get('confirmPassword');

  // Check if both controls have values and if they match
  if (password && confirmPassword && password.value !== confirmPassword.value) {
    // Return an error object if passwords don't match
    return { PasswordNoMatch: true };
  }

  // Return null if passwords match or if any of the controls are missing
  return null;
};
