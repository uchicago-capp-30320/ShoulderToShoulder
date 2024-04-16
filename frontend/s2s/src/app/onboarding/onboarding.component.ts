import { Component, OnInit } from '@angular/core';
import { PrimeNGConfig } from "primeng/api"; 

// services
import { UserService } from '../_services/user.service';

/**
 * OnboardingComponent
 * 
 * This component manages the onboarding process for new users.
 * It guides users through multiple pages to collect demographic information, 
 * preferences, event availability, and scenarios. Users can navigate between 
 * pages, submit the collected data, and view a confirmation dialog.
 * 
 * Example:
 * ```
 * <app-onboarding></app-onboarding>
 * ```
 */
@Component({
  selector: 'app-onboarding',
  templateUrl: './onboarding.component.html',
  styleUrl: './onboarding.component.css'
})
export class OnboardingComponent implements OnInit{
  page: number = 2;
  maxPage: number = 5;
  showConfirm: boolean = false;

  constructor(
    public userService: UserService,
    private primengConfig: PrimeNGConfig
  ) {}

  ngOnInit(): void {
    this.primengConfig.ripple = true;
    this.getProgressBarWidth();
  }

  /**
   * Updates the width of the progress bar.
   */
  getProgressBarWidth() {
    const progressBar = document.getElementById("progressBar");
    console.log('adjusting progress bar width...')
    if (progressBar) {
      console.log(`${((this.page-1) / (this.maxPage-1)) * 100}%`)
      progressBar.style.width = `${((this.page-1) / (this.maxPage-1)) * 100}%`;
    }
  }

  /**
   * Hides the progress bar.
   */
  hideProgressBar() {
    const progressBar = document.getElementById("progressIndicator");
    if (progressBar) {
      progressBar.style.display = "none";
    }
  }

  /**
   * Shows the progress bar.
   */
  showProgressBar() {
    const progressBar = document.getElementById("progressIndicator");
    if (progressBar) {
      progressBar.style.display = "block";
    }
  }

  /**
   * Moves to the next page.
   */
  nextPage() {
    this.getProgressBarWidth();
    this.goToTop();
    this.page++;
    this.getProgressBarWidth();
    if (this.page === 1) {
      this.hideProgressBar();
    } else {
      this.showProgressBar();
    }
  }

  /**
   * Moves to the previous page.
   */
  previousPage() {
    this.goToTop();
    this.page--;
    this.getProgressBarWidth();
    if (this.page === 1) {
      this.hideProgressBar();
    } else {
      this.showProgressBar();
    }
  }

  goToTop() {
    window.scrollTo({ 
      top: 0, 
      left: 0, 
      behavior: 'smooth' 
    });
  }

  showConfirmDialog() {
    this.showConfirm = true;
  }

  /**
   * Submits the demographics form.
   */
  onSubmit() {
    this.showConfirm = false;
    console.log("Onboading forms submitted")
    console.log("Demographics: ", this.userService.demographicsForm.value);
    console.log("Preferences: ", this.userService.preferencesForm.value);
    console.log("Event Availability: ", this.userService.eventAvailabilityForm.value);
    console.log("Scenarios: ", this.userService.scenariosForm.value);
    this.page = this.maxPage + 1;
  }

}
