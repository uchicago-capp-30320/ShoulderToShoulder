import { Component } from '@angular/core';

@Component({
  selector: 'app-onboarding',
  templateUrl: './onboarding.component.html',
  styleUrl: './onboarding.component.css'
})
export class OnboardingComponent {
  page: number = 1;

  constructor() {}

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

}
