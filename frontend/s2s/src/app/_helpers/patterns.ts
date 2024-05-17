// Purpose: This file contains regular expressions for survey form validation.

/**
 * Pattern for ensuring a srong password.
 */
export const StrongPasswordRegx: RegExp =
  /^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[!@#$%^&*()+\-=[\]{};':"\\|,.<>/?]).{8,}$/;

/**
 * Pattern for ensuring an input is only numbers.
 */
export const NumberRegx: RegExp = /^[0-9]*$/;
