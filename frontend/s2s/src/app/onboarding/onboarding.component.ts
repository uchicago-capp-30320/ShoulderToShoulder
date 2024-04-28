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
  showExit: boolean = false;
  invalidDialogMessage: string = "Please fill out all required fields.";
  progressBarColorMap: { [index: number]: string[] } = {
    1: ['#104C56', '#FFECD1'],
    2: ['#166A79', '#FFECD1'],
    3: ['#1C889B', '#FFECD1'],
    4: ['#23A6BE', '#001524'],
  }

  constructor(
    public onboardingService: OnboardingService,
    private primengConfig: PrimeNGConfig
  ) {
  }

  ngOnInit(): void {
    this.primengConfig.ripple = true;
    this.page = 0;
  }

  /**
   * Moves to the next page.
   */
  nextPage() {
    this.goToTop();
    this.page++;
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
      for (let control in form.controls) {
        let formControl = form.controls[control]
        if (formControl.invalid) {
          formControl.markAsDirty();
        }
      }
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
   * Shows the exit dialog.
   */
  showExitDialog() {
    this.showExit = true;
  }

  /**
   * Exist onboarding by sending current data to the backend and 
   * signing user out.
   */
  exitOnboarding() {
    this.onboardingService.exitOnboarding();
  }

  /**
   * Submits the demographics form.
   */
  onSubmit() {
    this.showConfirm = false;
    this.page = this.maxPage + 1;
    this.onboardingService.submitOnboardingForms()
    console.log("Onboading forms submitted")
  }

}
