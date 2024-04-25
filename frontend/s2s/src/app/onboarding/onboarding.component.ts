import { Component, OnInit } from '@angular/core';
import { FormGroup } from '@angular/forms';
import { PrimeNGConfig } from "primeng/api"; 

// services
import { OnboardingService } from '../_services/onboarding.service';
import { formControlFieldMap } from '../_helpers/preferences';

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
  page: number = 0;
  maxPage: number = 4;
  showConfirm: boolean = false;
  showInvalidDialog: boolean = false;
  invalidDialogMessage: string = "Please fill out all required fields.";

  constructor(
    public onboardingService: OnboardingService,
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

    // test - submitting availability form
    if ((this.page-1) == 3) {
      console.log(this.page)
      this.onboardingService.submitAvailabilityForm();
    }
  }

  nextButtonDisabled(){
    return (this.page === 1 && this.onboardingService.preferencesForm.invalid)
    || (this.page===2 && this.onboardingService.demographicsForm.invalid)
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

  /**
   * Highlights invalid fields on the current page.
   * 
   * @param event The click event.
   */
  highlightInvalidFields(event: any) {
    // map the page number to the form
    let pageFormMap: { [index: number]: FormGroup} = {
      1: this.onboardingService.preferencesForm,
      2: this.onboardingService.demographicsForm,
      4: this.onboardingService.scenariosForm
    }

    // if the button is disabled, the form is invalid
    if (event.target.querySelector('button') && event.target.querySelector('button').disabled){
      let form = pageFormMap[this.page];
      form.markAllAsTouched();
      this.showInvalidDialog = form.invalid ? true : false;
      this.invalidDialogMessage += " The following fields are missing values: "

      // get the missing fields
      let missingFields = [];
      for (let control in form.controls) {
        let formControl = form.controls[control]
        if (formControl.invalid) {
          missingFields.push(formControlFieldMap[control]);
        }
      }

      this.invalidDialogMessage += missingFields.join(", ");
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
    console.log("Demographics: ", this.onboardingService.demographicsForm.value);
    console.log("Preferences: ", this.onboardingService.preferencesForm.value);
    console.log("Event Availability: ", this.onboardingService.eventAvailabilityForm.value);
    console.log("Scenarios: ", this.onboardingService.scenariosForm.value);
    this.page = this.maxPage + 1;
  }

}
