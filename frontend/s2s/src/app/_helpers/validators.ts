import { AbstractControl, ValidationErrors, ValidatorFn } from '@angular/forms';

/**
 * Custom validator to check if two controls have the same value.
 * 
 * @param control Control to validate
 * @returns If the control is valid, returns null. If the control is invalid, 
 *          returns an error object.
 */
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
