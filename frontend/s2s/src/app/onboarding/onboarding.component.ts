import { Component, OnInit } from '@angular/core';
import { FormGroup } from '@angular/forms';
import { PrimeNGConfig } from "primeng/api"; 
import { Router } from '@angular/router';

// services
import { OnboardingService } from '../_services/onboarding.service';
import { AuthService } from '../_services/auth.service';

// helpers
import { formControlFieldMap } from '../_helpers/preferences';

/**
 * OnboardingComponent
 * 
 * This component manages the onboarding process for new users.
 * It guides users through multiple pages to collect demographic information, 
 * preferences, event availability, and scenarios. Users can navigate between 
 * pages, submit the collected data, and view a confirmation dialog.
 * 
 * @example
 * ```
 * <app-onboarding></app-onboarding>
 * ```
 * 
 * @see OnboardingService
 * @see AuthService
 */
@Component({
  selector: 'app-onboarding',
  templateUrl: './onboarding.component.html',
  styleUrl: './onboarding.component.css'
})
export class OnboardingComponent implements OnInit{
  page: number = -1;
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
    private primengConfig: PrimeNGConfig,
    private authService: AuthService,
    private router: Router
  ) {
  }

  ngOnInit(): void {
    this.primengConfig.ripple = true;
    
    // set page
    this.authService.getOnboardingStatus().subscribe(onboarded => {
      if (onboarded) {
        this.router.navigate(['/profile/1']);
      } else {
        this.page = 0;
      }
    });
  }

  /**
   * Moves to the next page of the onboarding process.
   */
  nextPage() {
    this.goToTop();
    this.page++;
  }

  /**
   * Determines if the next button should be disabled.
   * 
   * @returns True if the next button should be disabled, false otherwise.
   */
  nextButtonDisabled(){
    return (this.page === 1 && this.onboardingService.preferencesForm.invalid)
    || (this.page===2 && this.onboardingService.demographicsForm.invalid)
  }

  /**
   * Moves to the previous page of the onboarding process.
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
    this.invalidDialogMessage = '';
    
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
   * Hides the confirmation dialog.
   */
  hideConfirmDialog() {
    this.showConfirm = false;
    document.body.style.overflow = 'auto';
  }

  /**
   * Shows the exit dialog.
   */
  showExitDialog() {
    this.showExit = true;
  }

  /**
   * Hides the invalid dialog.
   */
  hideInvalidDialog() {
    this.showInvalidDialog = false;
    document.body.style.overflow = 'auto';
  }

  /**
   * Exits onboarding by sending current data to the backend and signing 
   * user out.
   */
  exitOnboarding() {
    let submit = this.page === this.maxPage + 1 ? true : false;
    this.onboardingService.exitOnboarding(submit);
  }

  /**
   * Submits the onboarding forms.
   */
  onSubmit() {
    this.showConfirm = false;
    this.page = this.maxPage + 1;
    this.onboardingService.submitOnboardingForms(true).subscribe(() => {
      console.log("Onboading forms submitted");
      this.router.navigate(['/profile/1']);
    });
  }
}
