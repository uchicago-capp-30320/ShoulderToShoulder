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

interface UploadEvent {
  originalEvent: Event;
  files: File[];
}

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

  openOnboardingDialog() {
    this.showOnboardingDialog = true;
  }

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

  resetUserForm() {
    this.updateUserInformationForm = new FormGroup({
      firstName: new FormControl(this.user.first_name, Validators.required),
      lastName: new FormControl(this.user.last_name, Validators.required),
      email: new FormControl(this.user.email, [Validators.required, Validators.email]),
    })
  }

  submitNewPassword() {
    console.log('submitting new password')
    let username = this.changePasswordForm.get('username')?.value;
    let currentPassword = this.changePasswordForm.get('currentPassword')?.value;
    let password = this.changePasswordForm.get('password')?.value;
    let confirmPassword = this.changePasswordForm.get('confirmPassword')?.value;

    let passwordChange: PasswordChange = {
      email: username ? username : '',
      current_password: currentPassword ? currentPassword : '',
      password: password ? password : '',
      confirm_password: confirmPassword ? confirmPassword : ''
    };

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

  submitUserUpdate() {
    console.log('submitting user update');
    let firstName = this.updateUserInformationForm.get('firstName')?.value;
    let lastName = this.updateUserInformationForm.get('lastName')?.value;
    let email = this.updateUserInformationForm.get('email')?.value;

    let userUpdate: UserUpdate = {
      first_name: firstName ? firstName : '',
      last_name: lastName ? lastName : '',
      email: email ? email : '',
      username: email ? email : ''
    };

    this.authService.updateUser(userUpdate).pipe(
      catchError(error => {
        console.error('Error updating user:', error);
        return EMPTY;
      })
    ).subscribe((response) => {
      // update user information
      let newUser = response as User;
      this.user = newUser;
      localStorage.setItem('user', JSON.stringify(newUser));
      this.authService.user.next(newUser);
      this.resetUserForm();

      console.log('User updated successfully!');
    });
  }

  submitOnboardingChanges() {
    console.log('submitting onboarding changes');
    this.showOnboardingDialog = false;
    this.onboardingService.updateOnboarding();
  }

  deleteAccount() {
    console.log('deleting account');
    this.showDeleteDialog = false;
    this.authService.deleteAccount(this.user);
    this.authService.logout();
  }

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

  uploadFileToServer(file: File) {
    this.profileService.uploadProfilePicture(file, this.user.id);
  }

  clearUpload(event: any) {
    event.files = []; // Clear the files array
    this.fileUpload.clear(); // Assuming `fileUpload` is a ViewChild reference to the p-fileUpload component
}

}
