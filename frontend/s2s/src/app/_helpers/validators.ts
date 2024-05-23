// Purpose: Contains custom validators for Angular forms.
import { AbstractControl, ValidationErrors, ValidatorFn } from '@angular/forms';

/**
 * Custom validator to check if a password meets the following criteria:
 * - Contains at least one uppercase letter
 * - Contains at least one lowercase letter
 * - Contains at least one number
 * - Contains at least one special character
 * @returns If the password meets all criteria, returns null. If the password
 *          does not meet all criteria, returns an error object.
 */
export function strongPasswordValidator(): ValidatorFn {
  return (control: AbstractControl): ValidationErrors | null => {
    const value = control.value;

    if (!value) {
      return null;
    }

    const errors: any = {};

    if (!/[A-Z]/.test(value)) {
      errors.missingUpperCase = 'Password must contain at least one uppercase letter.';
    }
    if (!/[a-z]/.test(value)) {
      errors.missingLowerCase = 'Password must contain at least one lowercase letter.';
    }
    if (!/\d/.test(value)) {
      errors.missingNumber = 'Password must contain at least one number.';
    }
    if (!/[!@#$%^&*()+\-=[\]{};':"\\|,.<>/?]/.test(value)) {
      errors.missingSpecial = 'Password must contain at least one special character.';
    }
    if (value.length < 8) {
      errors.minLength = 'Password must be at least 8 characters long.';
    }

    return Object.keys(errors).length ? errors : null;
  };
}

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
