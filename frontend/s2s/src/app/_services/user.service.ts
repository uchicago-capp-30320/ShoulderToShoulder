import { Injectable } from '@angular/core';
import { FormBuilder, FormGroup, Validators, FormControl } from '@angular/forms';
import { NumberRegx } from '../_helpers/patterns';
import { User } from '../_models/user';

// services
import { AuthService } from './auth.service';

@Injectable({
  providedIn: 'root'
})
export class UserService {
  loggedIn: boolean = false;
  user: User | null = null;

  constructor(
    private fb: FormBuilder,
    private authService: AuthService
    ) {
    this.setUser();
    }

  setUser() {
    this.authService.user.subscribe(user => {
      this.user = user;
      this.loggedIn = true;
    });
  }
}
