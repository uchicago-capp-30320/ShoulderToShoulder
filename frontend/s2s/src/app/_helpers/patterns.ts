// Definition: Regular expressions for validation
export const StrongPasswordRegx: RegExp =
  /^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[!@#$%^&*()+\-=[\]{};':"\\|,.<>/?]).{8,}$/;
