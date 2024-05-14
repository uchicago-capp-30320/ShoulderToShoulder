import { Component, OnInit, ViewChild } from '@angular/core';
import { FormGroup, FormControl, Validators } from '@angular/forms';
import { catchError } from 'rxjs/operators';
import { EMPTY } from 'rxjs';

// services
import { AuthService } from '../_services/auth.service';
import { OnboardingService } from '../_services/onboarding.service';
import { ProfileService } from '../_services/profile.service';

// helpers
import { StrongPasswordRegx } from '../_helpers/patterns';
import { confirmPasswordValidator, differentPasswordValidator } from '../_helpers/validators';
import { PasswordChange } from '../_models/password-change';
import { UserUpdate } from '../_models/user';
import { User } from '../_models/user';

/**
 * Component to display user profile settings.
 * 
 * This component displays the user's profile settings, including the ability to 
 * change their password, update their user information, upload a profile picture, 
 * and delete their account.
 * 
 * @example
 * ```
 * <app-profile-settings></app-profile-settings>
 * ```
 * 
 * @remarks
 * This component relies on the AuthService, OnboardingService, and ProfileService 
 * to manage user profile settings.
 * 
 * @see AuthService
 * @see OnboardingService
 * @see ProfileService
 * 
 * @export
 * @class ProfileSettingsComponent
 */
@Component({
  selector: 'app-profile-settings',
  templateUrl: './profile-settings.component.html',
  styleUrl: './profile-settings.component.css'
})
export class ProfileSettingsComponent implements OnInit {
  showOnboardingDialog = false;
  showDeleteDialog = false;
  user = this.authService.userValue;
  profilePictureUrl: string = '';
  reader = new FileReader(); 
  uploadedFiles: any[] = [];
  showInvalidPassword = false;
  @ViewChild('fileUpload') fileUpload: any;

  // form used to change user password
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

  // form used to update user information
  updateUserInformationForm = new FormGroup({
    firstName: new FormControl(this.user.first_name, Validators.required),
    lastName: new FormControl(this.user.last_name, Validators.required),
    email: new FormControl(this.user.email, [Validators.required, Validators.email]),
  });

  showInvalidPasswordMessage = "Invalid password. Please try again."

  constructor(
    private authService: AuthService,
    private onboardingService: OnboardingService,
    public profileService: ProfileService
  ) {}

  ngOnInit(): void {
    this.profileService.profilePicture.subscribe(profilePictureUrl => {
      this.profilePictureUrl = profilePictureUrl;
    });
  }

  /**
   * Opens the onboarding dialog.
   */
  openOnboardingDialog() {
    this.showOnboardingDialog = true;
  }

  /**
   * Opens the delete dialog.
   */
  openDeleteDialog() {
    this.showDeleteDialog = true;
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
   * Resets the user form.
   */
  resetUserForm() {
    this.updateUserInformationForm = new FormGroup({
      firstName: new FormControl(this.user.first_name, Validators.required),
      lastName: new FormControl(this.user.last_name, Validators.required),
      email: new FormControl(this.user.email, [Validators.required, Validators.email]),
    })
  }

  /**
   * Submits the new password.
   */
  submitNewPassword() {
    console.log('submitting new password')
    let username = this.changePasswordForm.get('username')?.value;
    let currentPassword = this.changePasswordForm.get('currentPassword')?.value;
    let password = this.changePasswordForm.get('password')?.value;
    let confirmPassword = this.changePasswordForm.get('confirmPassword')?.value;

    // create password change object
    let passwordChange: PasswordChange = {
      email: username ? username : '',
      current_password: currentPassword ? currentPassword : '',
      password: password ? password : '',
      confirm_password: confirmPassword ? confirmPassword : ''
    };

    // change password
    this.authService.changePassword(passwordChange).pipe(
      catchError(error => {
        console.error('Error changing password:', error);
        if (error.status == 400) {
          this.showInvalidPassword = true;
        }
        return EMPTY;
      })
    ).subscribe(() => {
      console.log('Password changed successfully');
      this.changePasswordForm.reset();
      this.authService.logout();
    });
  }

  /**
   * Submits the user update.
   */
  submitUserUpdate() {
    console.log('submitting user update');
    let firstName = this.updateUserInformationForm.get('firstName')?.value;
    let lastName = this.updateUserInformationForm.get('lastName')?.value;
    let email = this.updateUserInformationForm.get('email')?.value;

    // create user update object
    let userUpdate: UserUpdate = {
      first_name: firstName ? firstName : '',
      last_name: lastName ? lastName : '',
      email: email ? email : '',
      username: email ? email : ''
    };

    // update user information
    this.authService.updateUser(userUpdate).pipe(
      catchError(error => {
        console.error('Error updating user:', error);
        return EMPTY;
      })
    ).subscribe((response) => {
      // update user information
      let newUser = response as User;
      this.user = newUser;
      sessionStorage.setItem('user', JSON.stringify(newUser));
      this.authService.user.next(newUser);
      this.resetUserForm();

      console.log('User updated successfully!');
    });
  }

  /**
   * Submits the onboarding changes.
   */
  submitOnboardingChanges() {
    console.log('submitting onboarding changes');
    this.showOnboardingDialog = false;
    this.onboardingService.updateOnboarding();
  }

  /**
   * Cancels the onboarding changes.
   */
  cancelOnboardingChanges() {
    console.log('canceling onboarding changes');
    this.showOnboardingDialog = false;
    this.onboardingService.cancelOnboarding();
  }

  /**
   * Deletes the user account.
   */
  deleteAccount() {
    console.log('deleting account');
    this.showDeleteDialog = false;
    this.authService.deleteAccount(this.user).subscribe(() => {
      this.authService.logout();
      console.log('Account deleted successfully');
    })
  }

  /**
   * Handles the file upload event.
   * 
   * @param event The file upload event.
   */
  onUpload(event: any) {
    if (event.files && event.files[0]) {
      const file = event.files[0];
      var reader = new FileReader();
  
      reader.readAsDataURL(file); // Read file as data URL for preview purposes
  
      reader.onload = (e) => { // Called once readAsDataURL is completed
        this.profilePictureUrl = e.target?.result as string;  // Update image preview
        this.uploadFileToServer(file);  // Call function to upload file to server
      }

      this.clearUpload(event);
    }
  }

  /**
   * Uploads a file to the server.
   * 
   * @param file The file to upload.
   */
  uploadFileToServer(file: File) {
    this.profileService.uploadProfilePicture(file, this.user.id);
  }

  /**
   * Clears the file upload.
   * 
   * @param event The file upload event.
   */
  clearUpload(event: any) {
    event.files = []; // Clear the files array
    this.fileUpload.clear(); // Assuming `fileUpload` is a ViewChild reference to the p-fileUpload component
  }
}
