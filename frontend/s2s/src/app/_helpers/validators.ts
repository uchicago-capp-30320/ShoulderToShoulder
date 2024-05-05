// Purpose: Contains custom validators for Angular forms.
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

  // check if both controls have values and if they are different
  if (password && confirmPassword && password.value !== confirmPassword.value) {
    return { PasswordNoMatch: true };
  }

  return null;
};

/**
 * Custom validator to ensure that two controls have different values.
 * The primary use case is to ensure that the new password is different from the 
 * current password.
 * 
 * @param control Control to validate
 * @returns If the control is valid, returns null. If the control is invalid,
 *         returns an error object.
 */
export const differentPasswordValidator: ValidatorFn = (control: AbstractControl): ValidationErrors | null => {
  const currentPassword = control.get('currentPassword');
  const password = control.get('password');

  // check if both controls have values and if they match
  if (currentPassword && password && currentPassword.value === password.value) {
    return { PasswordMatch: true };
  }

  return null;
}
