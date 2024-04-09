import { Component } from '@angular/core';
import { UserService } from '../_services/user.service';

@Component({
  selector: 'app-onboarding',
  templateUrl: './onboarding.component.html',
  styleUrl: './onboarding.component.css'
})
export class OnboardingComponent {
  page: number = 2; // FIXME: Change to 1

  constructor(
    public userService: UserService
  ) {}

  /**
   * Moves to the next page.
   */
  nextPage() {
    this.page++;
  }

  /**
   * Moves to the previous page.
   */
  previousPage() {
    this.page--;
  }

  /**
   * Submits the demographics form.
   */
  onSubmit() {
    console.log("Onboading forms submitted")
  }

}
