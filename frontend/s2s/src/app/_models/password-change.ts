// Purpose: This file contains the interface for the password change object.
export interface PasswordChange {
    email: string;
    current_password: string;
    password: string;
    confirm_password: string;
}
