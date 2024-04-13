import { Component } from '@angular/core';
import { UserService } from '../_services/user.service';

@Component({
  selector: 'app-onboarding',
  templateUrl: './onboarding.component.html',
  styleUrl: './onboarding.component.css'
})
export class OnboardingComponent {
  page: number = 3; // FIXME: Change to 1
  maxPage: number = 5;

  constructor(
    public userService: UserService
  ) {}

  /**
   * Moves to the next page.
   */
  nextPage() {
    this.gotoTop();
    this.page++;
  }

  /**
   * Moves to the previous page.
   */
  previousPage() {
    this.gotoTop();
    this.page--;
  }

  gotoTop() {
    window.scroll({ 
      top: 0, 
      left: 0, 
      behavior: 'smooth' 
    });
  }

  /**
   * Submits the demographics form.
   */
  onSubmit() {
    console.log("Onboading forms submitted")
  }

}
