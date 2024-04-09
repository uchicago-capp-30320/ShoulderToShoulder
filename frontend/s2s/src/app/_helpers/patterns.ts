// Definition: Regular expressions for validation
export const StrongPasswordRegx: RegExp =
  /^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[!@#$%^&*()+\-=[\]{};':"\\|,.<>/?]).{8,}$/;

export const NumberRegx: RegExp = /^[0-9]*$/;