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
  maxPage: number = 4;
  showConfirm: boolean = false;
  showInvalidDialog: boolean = false;

  constructor(
    public userService: UserService,
    private primengConfig: PrimeNGConfig
  ) {}

  ngOnInit(): void {
    this.primengConfig.ripple = true;
  }

  /**
   * Moves to the next page.
   */
  nextPage() {
    this.goToTop();
    this.page++;
  }

  /**
   * Moves to the previous page.
   */
  previousPage() {
    this.goToTop();
    this.page--;
  }

  /**
   * Scrolls to the top of the page.
   */
  goToTop() {
    window.scrollTo({ 
      top: 0, 
      left: 0, 
      behavior: 'smooth' 
    });
  }

  highlightInvalidFields(event: any) {
    if (event.target.querySelector('button').disabled){
      if (this.page == 1) {
        this.userService.preferencesForm.markAllAsTouched();
        this.showInvalidDialog = this.userService.preferencesForm.invalid ? true : false;
      } else if (this.page == 2) {
        this.userService.demographicsForm.markAllAsTouched();
        console.log(this.userService.demographicsForm)
        this.showInvalidDialog = this.userService.demographicsForm.invalid ? true : false;
      } else if (this.page == 3) {
        this.userService.eventAvailabilityForm.markAllAsTouched();
        this.showInvalidDialog = this.userService.eventAvailabilityForm.invalid ? true : false;
      } else if (this.page == 4) {
        this.userService.scenariosForm.markAllAsTouched();
        this.showInvalidDialog = this.userService.scenariosForm.invalid ? true : false;
      } 
    }
  }

  /**
   * Shows the confirmation dialog.
   */
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
